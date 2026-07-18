"""Launch LeRobot policy training on Modal.

Examples:
    modal run training/main.py --dataset ./my_dataset --gpu H200 --policy act --steps 50000
    modal run training/main.py --dataset user/my_dataset --gpu a100 --policy groot --steps 20000
    python training/main.py --dataset user/my_dataset --gpu B200 --policy molmoact2 --steps 10000 --lr 1e-5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
LEROBOT_ROOT = REPO_ROOT / "thirdparty" / "lerobot"

# Modal workers always run Linux, regardless of the local (submitting) OS, so
# remote paths must stay POSIX even when this script runs on Windows.
REMOTE_DATA_ROOT = PurePosixPath("/data")
REMOTE_OUTPUT_ROOT = PurePosixPath("/outputs")

DEFAULT_STEPS = 100_000
DEFAULT_GPU = "A100"
DEFAULT_POLICY = "act"

# Modal GPU shortcodes. User-facing aliases are case-insensitive.
GPU_ALIASES: dict[str, str] = {
    "t4": "T4",
    "l4": "L4",
    "a10": "A10",
    "l40s": "L40S",
    "a100": "A100",
    "a100-40gb": "A100-40GB",
    "a100-80gb": "A100-80GB",
    "rtx-pro-6000": "RTX-PRO-6000",
    "h100": "H100",
    "h100!": "H100!",
    "h200": "H200",
    # "b100" is accepted as a friendly alias for Blackwell B200 (Modal has no B100).
    "b100": "B200",
    "b200": "B200",
    "b200+": "B200+",
    "b300": "B300",
}

KNOWN_POLICIES = (
    "act",
    "diffusion",
    "groot",
    "molmoact2",
    "smolvla",
    "pi0",
    "pi05",
    "pi0_fast",
    "vqbet",
    "tdmpc",
    "xvla",
    "wall_x",
    "vla_jepa",
    "multi_task_dit",
    "eo1",
    "evo1",
    "lingbot_va",
    "fastwam",
    "gaussian_actor",
)

app = modal.App("soccerbot-train")

dataset_volume = modal.Volume.from_name("soccerbot-train-datasets", create_if_missing=True)
output_volume = modal.Volume.from_name("soccerbot-train-outputs", create_if_missing=True)

train_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("ffmpeg", "git", "build-essential", "libgl1", "libglib2.0-0")
    .pip_install("uv")
    .add_local_dir(
        str(LEROBOT_ROOT),
        remote_path="/opt/lerobot",
        copy=True,
        ignore=[
            "**/.git/**",
            "**/.venv/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/tests/**",
            "**/docs/**",
            "**/benchmarks/**",
            "**/examples/**",
            "**/.pytest_cache/**",
            "**/.ruff_cache/**",
            "**/.mypy_cache/**",
        ],
    )
    .run_commands(
        "cd /opt/lerobot && uv pip install --system -e '.[dataset,training,groot,molmoact2]'",
    )
)


@dataclass(frozen=True)
class DatasetSpec:
    """Resolved dataset identity for LeRobot CLI args."""

    repo_id: str
    root: str | None  # path on the machine that will run training
    is_local: bool
    local_path: Path | None = None


@dataclass(frozen=True)
class TrainRequest:
    dataset: DatasetSpec
    policy: str
    gpu: str
    steps: int
    lr: float | None = None
    batch_size: int | None = None
    seed: int | None = None
    output_dir: str | None = None
    job_name: str | None = None
    policy_repo_id: str | None = None
    push_to_hub: bool = False
    wandb: bool = False
    num_workers: int | None = None
    save_freq: int | None = None
    log_freq: int | None = None
    only_action_expert_ft: bool = False
    extra_args: tuple[str, ...] = ()


def normalize_gpu(gpu: str) -> str:
    """Normalize a user GPU string to a Modal GPU shortcode."""
    raw = gpu.strip()
    if not raw:
        raise ValueError("GPU must be a non-empty string (e.g. H200, A100, B200).")

    # Allow count suffixes like "H100:2" / "a100-80gb:4".
    count_suffix = ""
    base = raw
    if ":" in raw:
        base, count = raw.rsplit(":", 1)
        if not count.isdigit() or int(count) < 1:
            raise ValueError(f"Invalid GPU count in {gpu!r}; expected e.g. H100:2.")
        count_suffix = f":{count}"

    key = base.strip().lower()
    if key in GPU_ALIASES:
        return GPU_ALIASES[key] + count_suffix

    # Accept already-canonical Modal names (and unknown future shortcodes).
    cleaned = base.strip()
    suffix_bang = cleaned.endswith("!")
    suffix_plus = cleaned.endswith("+")
    body = cleaned.rstrip("!+")
    canonical = "-".join(
        (p.upper() if not p.lower().endswith("gb") else p[:-2] + "GB") for p in body.split("-")
    )
    if suffix_bang:
        canonical += "!"
    if suffix_plus:
        canonical += "+"
    return canonical + count_suffix


def resolve_dataset(dataset: str) -> DatasetSpec:
    """Resolve ``./path`` or ``username/dataset`` into LeRobot dataset args."""
    raw = dataset.strip()
    if not raw:
        raise ValueError("--dataset must be a local path or a Hub id like username/dataset.")

    path = Path(raw).expanduser()
    explicit_path = raw.startswith((".", "~", "/")) or path.exists()

    if explicit_path:
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Local dataset path does not exist: {resolved}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Local dataset path is not a directory: {resolved}")
        return DatasetSpec(
            repo_id=resolved.name,
            root=str(resolved),
            is_local=True,
            local_path=resolved,
        )

    parts = raw.split("/")
    if len(parts) >= 2 and all(parts):
        return DatasetSpec(repo_id=raw, root=None, is_local=False, local_path=None)

    raise ValueError(
        f"Invalid dataset {dataset!r}. Use a local path (./dataset) "
        "or a Hub id (username/dataset)."
    )


def build_lerobot_args(req: TrainRequest) -> list[str]:
    """Build ``lerobot-train`` CLI args. Omits optional overrides when unset."""
    if req.policy not in KNOWN_POLICIES:
        print(
            f"Warning: policy {req.policy!r} is not in the known list "
            f"{KNOWN_POLICIES}; passing through to lerobot-train.",
            file=sys.stderr,
        )

    push_to_hub = req.push_to_hub or bool(req.policy_repo_id)
    job_name = req.job_name or f"{req.policy}_{req.dataset.repo_id.replace('/', '_')}"

    args = [
        f"--dataset.repo_id={req.dataset.repo_id}",
        f"--policy.type={req.policy}",
        f"--steps={req.steps}",
        "--policy.device=cuda",
        f"--policy.push_to_hub={'true' if push_to_hub else 'false'}",
        f"--wandb.enable={'true' if req.wandb else 'false'}",
        f"--job_name={job_name}",
    ]

    if req.dataset.root is not None:
        args.append(f"--dataset.root={req.dataset.root}")

    if req.lr is not None:
        args.append(f"--policy.optimizer_lr={req.lr}")

    if req.batch_size is not None:
        args.append(f"--batch_size={req.batch_size}")

    if req.seed is not None:
        args.append(f"--seed={req.seed}")

    if req.output_dir is not None:
        args.append(f"--output_dir={req.output_dir}")

    if req.policy_repo_id is not None:
        args.append(f"--policy.repo_id={req.policy_repo_id}")

    if req.num_workers is not None:
        args.append(f"--num_workers={req.num_workers}")

    if req.save_freq is not None:
        args.append(f"--save_freq={req.save_freq}")

    if req.log_freq is not None:
        args.append(f"--log_freq={req.log_freq}")

    if req.policy == "molmoact2":
        # Default finetune mode is LoRA on the VLM; --onlyactionexpertft switches
        # to freezing everything except the action expert (requires continuous mode).
        if req.only_action_expert_ft:
            args.extend(
                [
                    "--policy.train_action_expert_only=true",
                    "--policy.enable_lora_vlm=false",
                    "--policy.action_mode=continuous",
                ]
            )
        else:
            args.append("--policy.enable_lora_vlm=true")

    args.extend(req.extra_args)
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a LeRobot training run to Modal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Local dataset directory (./dataset) or Hugging Face Hub id (username/dataset).",
    )
    parser.add_argument(
        "--gpu",
        default=DEFAULT_GPU,
        help="Modal GPU shortcode or alias (H200, A100, B200, b100, A100-80GB, H100:2, ...).",
    )
    parser.add_argument(
        "--policy",
        default=DEFAULT_POLICY,
        help=f"Policy type for --policy.type=... Known: {', '.join(KNOWN_POLICIES)}.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Number of training steps (LeRobot default is 100000).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Optional learning-rate override (--policy.optimizer_lr). Default: policy preset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override. Default: LeRobot/policy default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional training seed override.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory inside the Modal container (default under /outputs).",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Optional LeRobot job name.",
    )
    parser.add_argument(
        "--policy-repo-id",
        default=None,
        help="Optional Hub repo id for the trained policy (enables push_to_hub).",
    )
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Push the trained policy to the Hub (requires --policy-repo-id).",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional dataloader worker override.",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=None,
        help="Optional checkpoint save frequency override.",
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=None,
        help="Optional logging frequency override.",
    )
    parser.add_argument(
        "--onlyactionexpertft",
        dest="only_action_expert_ft",
        action="store_true",
        default=False,
        help=(
            "MolmoAct2 only: finetune the action expert only "
            "(sets train_action_expert_only + action_mode=continuous, disables LoRA). "
            "Default molmoact2 mode keeps enable_lora_vlm=true."
        ),
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra raw lerobot-train CLI arg (repeatable), e.g. --extra-arg=--policy.use_amp=true",
    )
    return parser.parse_args(argv)


def request_from_args(args: argparse.Namespace) -> TrainRequest:
    dataset = resolve_dataset(args.dataset)
    gpu = normalize_gpu(args.gpu)

    output_dir = args.output_dir
    if output_dir is None:
        safe_name = dataset.repo_id.replace("/", "_")
        output_dir = str(REMOTE_OUTPUT_ROOT / f"{args.policy}_{safe_name}")

    push_to_hub = bool(args.push_to_hub or args.policy_repo_id)
    if args.push_to_hub and not args.policy_repo_id:
        raise ValueError("--push-to-hub requires --policy-repo-id.")
    if args.only_action_expert_ft and args.policy != "molmoact2":
        raise ValueError("--onlyactionexpertft is only supported with --policy molmoact2.")

    return TrainRequest(
        dataset=dataset,
        policy=args.policy,
        gpu=gpu,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=output_dir,
        job_name=args.job_name,
        policy_repo_id=args.policy_repo_id,
        push_to_hub=push_to_hub,
        wandb=args.wandb,
        num_workers=args.num_workers,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        only_action_expert_ft=args.only_action_expert_ft,
        extra_args=tuple(args.extra_arg or ()),
    )


def upload_local_dataset(local_path: Path) -> str:
    """Upload a local LeRobot dataset directory into the shared Modal volume."""
    remote_rel = f"datasets/{local_path.name}"
    print(f"Uploading local dataset {local_path} -> volume:/{remote_rel} ...")
    with dataset_volume.batch_upload(force=True) as batch:
        batch.put_directory(str(local_path), remote_rel)
    return str(REMOTE_DATA_ROOT / remote_rel)


def secrets_for_run(req: TrainRequest) -> list[modal.Secret]:
    """Forward local HF/W&B tokens into the Modal container when present."""
    env: dict[str, str] = {}
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if key in os.environ and os.environ[key]:
            env[key] = os.environ[key]
    if req.wandb:
        for key in ("WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY"):
            if key in os.environ and os.environ[key]:
                env[key] = os.environ[key]
    if not env:
        return []
    return [modal.Secret.from_dict(env)]


@app.function(
    image=train_image,
    timeout=24 * 60 * 60,
    memory=8192,
    volumes={
        str(REMOTE_DATA_ROOT): dataset_volume,
        str(REMOTE_OUTPUT_ROOT): output_volume,
    },
)
def run_training(cli_args: list[str]) -> int:
    """Run ``lerobot-train`` on a Modal GPU worker."""
    cmd = ["lerobot-train", *cli_args]
    print("Running:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(REMOTE_OUTPUT_ROOT / "hf"))
    env.setdefault("HF_LEROBOT_HOME", str(REMOTE_OUTPUT_ROOT / "lerobot"))
    completed = subprocess.run(cmd, check=False, env=env)
    output_volume.commit()
    if completed.returncode != 0:
        raise RuntimeError(f"lerobot-train failed with exit code {completed.returncode}")
    return completed.returncode


def submit_training(req: TrainRequest) -> int:
    """Prepare dataset uploads and submit the Modal training function."""
    if req.dataset.is_local:
        assert req.dataset.local_path is not None
        remote_root = upload_local_dataset(req.dataset.local_path)
        req = replace(
            req,
            dataset=DatasetSpec(
                repo_id=req.dataset.repo_id,
                root=remote_root,
                is_local=True,
                local_path=req.dataset.local_path,
            ),
        )

    cli_args = build_lerobot_args(req)
    print(f"Submitting Modal training run on gpu={req.gpu}")
    print(f"Policy={req.policy} steps={req.steps} dataset={req.dataset.repo_id}")
    if req.lr is None:
        print("Learning rate: policy default preset")
    else:
        print(f"Learning rate override: {req.lr}")

    fn = run_training.with_options(gpu=req.gpu, secrets=secrets_for_run(req))
    return fn.remote(cli_args)


@app.local_entrypoint()
def main(*arglist: str) -> None:
    """Entrypoint for ``modal run training/main.py --dataset ...``."""
    # Parse first so ``--help`` works without contacting Modal.
    req = request_from_args(parse_args(list(arglist)))
    raise SystemExit(submit_training(req))


if __name__ == "__main__":
    # Direct: python training/main.py --dataset ... --gpu H200 --policy act --steps 10000
    # Modal CLI imports this module (not as __main__) and calls ``main`` above.
    req = request_from_args(parse_args(sys.argv[1:]))
    with app.run():
        raise SystemExit(submit_training(req))
