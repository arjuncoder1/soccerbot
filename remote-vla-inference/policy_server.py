"""Modal LeRobot async PolicyServer for remote VLA inference.

Policy weights are chosen by the client on handshake (default: G1 π0.5 box-move).

  modal run remote-vla-inference/policy_server.py --gpu A100 --hours 2

Or:  ./remote-vla-inference/run_server.sh
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
LEROBOT_ROOT = REPO_ROOT / "thirdparty" / "lerobot"

GPU_ALIASES: dict[str, str] = {
    "t4": "T4",
    "l4": "L4",
    "a10": "A10",
    "l40s": "L40S",
    "a100": "A100",
    "a100-40gb": "A100-40GB",
    "a100-80gb": "A100-80GB",
    "h100": "H100",
    "h200": "H200",
    "b200": "B200",
}

DEFAULT_GPU = "A100"
DEFAULT_PORT = 8080
DEFAULT_HOURS = 2.0
# async gRPC + π0 / π0.5 stack
POLICY_SERVER_EXTRAS = "async,pi"

# Default checkpoint advertised in logs (client still sends the path).
DEFAULT_POLICY_ID = "sudoping01/pi05_g1_boxmove_v2"

app = modal.App("remote-vla-inference")

policy_server_image = (
    modal.Image.debian_slim(python_version="3.12")
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
        f"cd /opt/lerobot && uv pip install --system -e '.[{POLICY_SERVER_EXTRAS}]'",
    )
)


def normalize_gpu(gpu: str) -> str:
    raw = gpu.strip()
    if not raw:
        raise ValueError("GPU must be a non-empty string (e.g. A100, H100).")
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
    return base.strip() + count_suffix


# Ungated PaliGemma tokenizer mirror (google/paligemma-3b-pt-224 is gated).
PALIGEMMA_TOKENIZER = "leo009/paligemma-3b-pt-224"


def _serve_forever(host: str, port: int, fps: int) -> None:
    """Run PolicyServer with π0.5 tokenizer override (avoid gated google repo)."""
    import logging
    from concurrent import futures
    from dataclasses import asdict
    from pprint import pformat

    import grpc

    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.policies import get_policy_class, make_pre_post_processors
    from lerobot.transport import services_pb2, services_pb2_grpc

    cfg = PolicyServerConfig(host=host, port=port, fps=fps)
    logging.info(pformat(asdict(cfg)))

    class PiFriendlyPolicyServer(PolicyServer):
        def SendPolicyInstructions(self, request, context):  # noqa: N802
            import pickle
            import time

            from lerobot.async_inference.constants import SUPPORTED_POLICIES
            from lerobot.async_inference.helpers import RemotePolicyConfig

            if not self.running:
                self.logger.warning("Server is not running. Ignoring policy instructions.")
                return services_pb2.Empty()

            policy_specs = pickle.loads(request.data)  # nosec
            if not isinstance(policy_specs, RemotePolicyConfig):
                raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")
            if policy_specs.policy_type not in SUPPORTED_POLICIES:
                raise ValueError(
                    f"Policy type {policy_specs.policy_type} not supported. "
                    f"Supported policies: {SUPPORTED_POLICIES}"
                )

            self.logger.info(
                f"Receiving policy instructions | type={policy_specs.policy_type} | "
                f"path={policy_specs.pretrained_name_or_path} | device={policy_specs.device}"
            )

            # Cache: skip the ~200s rebuild if the same checkpoint is already
            # loaded on the same device. Common case: user Ctrl+C's the client
            # and reconnects -- no need to re-instantiate PaliGemma-3b + reload
            # 7GB of safetensors when nothing about the policy changed.
            cache_key = (
                policy_specs.policy_type,
                policy_specs.pretrained_name_or_path,
                policy_specs.device,
            )
            if (
                getattr(self, "_loaded_cache_key", None) == cache_key
                and getattr(self, "policy", None) is not None
            ):
                self.logger.info(
                    f"Reusing cached policy ({policy_specs.pretrained_name_or_path} "
                    f"on {policy_specs.device}); skipping rebuild."
                )
                # Client-provided metadata may change even when the checkpoint
                # doesn't; keep it in sync so feature routing / chunk sizing use
                # the fresh values.
                self.lerobot_features = policy_specs.lerobot_features
                self.actions_per_chunk = policy_specs.actions_per_chunk
                return services_pb2.Empty()

            self.device = policy_specs.device
            self.policy_type = policy_specs.policy_type
            self.lerobot_features = policy_specs.lerobot_features
            self.actions_per_chunk = policy_specs.actions_per_chunk

            policy_class = get_policy_class(self.policy_type)
            start = time.perf_counter()

            # The sudoping01/pi05_g1_boxmove_v2 checkpoint ships config
            # compile_model=True with compile_mode="max-autotune". On this Modal
            # image (torch 2.11) that TorchInductor compile stalls for minutes and
            # then SIGSEGVs, so inference never returns. Force eager execution:
            # a single A100 inference is a few seconds and is rock-solid.
            from lerobot.configs.policies import PreTrainedConfig

            policy_config = PreTrainedConfig.from_pretrained(policy_specs.pretrained_name_or_path)
            if getattr(policy_config, "compile_model", False):
                self.logger.info(
                    f"Disabling torch.compile (was compile_model=True, "
                    f"mode={getattr(policy_config, 'compile_mode', '?')}) to avoid Inductor hang/segfault."
                )
                policy_config.compile_model = False
            self.policy = policy_class.from_pretrained(
                policy_specs.pretrained_name_or_path, config=policy_config
            )
            self.policy.to(self.device)

            device_override = {"device": self.device}
            # Override gated google/paligemma tokenizer with a public mirror.
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config,
                pretrained_path=policy_specs.pretrained_name_or_path,
                preprocessor_overrides={
                    "device_processor": device_override,
                    "rename_observations_processor": {"rename_map": policy_specs.rename_map},
                    "tokenizer_processor": {"tokenizer_name": PALIGEMMA_TOKENIZER},
                },
                postprocessor_overrides={"device_processor": device_override},
            )
            self.logger.info(
                f"Policy ready on {self.device} in {time.perf_counter() - start:.1f}s "
                f"(tokenizer={PALIGEMMA_TOKENIZER})"
            )
            self._loaded_cache_key = cache_key
            return services_pb2.Empty()

    policy_server = PiFriendlyPolicyServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()


@app.function(
    image=policy_server_image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 4,
)
def serve_remote(port: int = DEFAULT_PORT, fps: int = 30, hours: float = DEFAULT_HOURS):
    host = "0.0.0.0"
    server_thread = threading.Thread(
        target=_serve_forever,
        kwargs={"host": host, "port": port, "fps": fps},
        daemon=True,
        name="lerobot-policy-server",
    )
    server_thread.start()
    time.sleep(2.0)

    with modal.forward(port, unencrypted=True) as tunnel:
        host_pub, port_pub = tunnel.tcp_socket
        print("=" * 72)
        print("Remote VLA PolicyServer is up on Modal.")
        print(f"  Internal bind : {host}:{port}")
        print(f"  Public TCP    : {host_pub}:{port_pub}")
        print()
        print("Point the G1 client at:")
        print(f"  --server_address={host_pub}:{port_pub}")
        print()
        print(f"Default policy (sent by client): {DEFAULT_POLICY_ID}")
        print(f"Keeping alive for ~{hours}h (Ctrl+C to stop).")
        print("=" * 72)
        sys.stdout.flush()

        deadline = time.time() + hours * 3600
        while time.time() < deadline:
            if not server_thread.is_alive():
                raise RuntimeError("PolicyServer thread exited unexpectedly")
            time.sleep(5.0)

    print("Tunnel closed; exiting.")


@app.local_entrypoint()
def main(
    gpu: str = DEFAULT_GPU,
    hours: float = DEFAULT_HOURS,
    port: int = DEFAULT_PORT,
    fps: int = 30,
):
    gpu_norm = normalize_gpu(gpu)
    timeout_s = max(60, int(hours * 3600) + 120)
    print(f"Launching PolicyServer on Modal gpu={gpu_norm} for {hours}h ...")
    serve_remote.with_options(gpu=gpu_norm, timeout=timeout_s).remote(
        port=port, fps=fps, hours=hours
    )
