"""Unit tests for Modal training CLI helpers (no Modal network calls)."""

from __future__ import annotations

from pathlib import Path

import pytest

from embodiment_g1_revo2 import STATE_ACTION_DIM, STATE_ACTION_NAMES
from main import (
    TrainRequest,
    build_lerobot_args,
    normalize_gpu,
    parse_args,
    request_from_args,
    resolve_dataset,
)


def test_normalize_gpu_aliases() -> None:
    assert normalize_gpu("h200") == "H200"
    assert normalize_gpu("H200") == "H200"
    assert normalize_gpu("a100") == "A100"
    assert normalize_gpu("A100-80GB") == "A100-80GB"
    assert normalize_gpu("a100-80gb") == "A100-80GB"
    assert normalize_gpu("b100") == "B200"
    assert normalize_gpu("b200") == "B200"
    assert normalize_gpu("H100:2") == "H100:2"
    assert normalize_gpu("a100-80gb:4") == "A100-80GB:4"


def test_normalize_gpu_rejects_bad_count() -> None:
    with pytest.raises(ValueError, match="Invalid GPU count"):
        normalize_gpu("H100:zero")


def test_resolve_hub_dataset() -> None:
    spec = resolve_dataset("alice/soccer-demos")
    assert spec.repo_id == "alice/soccer-demos"
    assert spec.root is None
    assert spec.is_local is False


def test_resolve_local_dataset(tmp_path: Path) -> None:
    ds = tmp_path / "my_dataset"
    ds.mkdir()
    (ds / "meta").mkdir()

    spec = resolve_dataset(str(ds))
    assert spec.is_local is True
    assert spec.repo_id == "my_dataset"
    assert spec.root == str(ds.resolve())
    assert spec.local_path == ds.resolve()


def test_resolve_local_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ds = Path("local_ds")
    ds.mkdir()
    spec = resolve_dataset("./local_ds")
    assert spec.is_local is True
    assert spec.repo_id == "local_ds"
    assert spec.local_path == (tmp_path / "local_ds").resolve()


def test_resolve_dataset_errors() -> None:
    with pytest.raises(ValueError, match="local path"):
        resolve_dataset("not-a-hub-id")
    with pytest.raises(FileNotFoundError):
        resolve_dataset("./definitely-missing-dataset-dir")


def test_build_lerobot_args_defaults_omit_overrides() -> None:
    req = TrainRequest(
        dataset=resolve_dataset("user/ds"),
        policy="act",
        gpu="H200",
        steps=50_000,
    )
    args = build_lerobot_args(req)
    assert "--dataset.repo_id=user/ds" in args
    assert "--policy.type=act" in args
    assert "--steps=50000" in args
    assert "--policy.device=cuda" in args
    assert "--policy.push_to_hub=false" in args
    assert all(not a.startswith("--policy.optimizer_lr=") for a in args)
    assert all(not a.startswith("--batch_size=") for a in args)
    assert all(not a.startswith("--dataset.root=") for a in args)


def test_build_lerobot_args_optional_overrides(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    ds.mkdir()
    req = TrainRequest(
        dataset=resolve_dataset(str(ds)),
        policy="groot",
        gpu="A100",
        steps=10_000,
        lr=1e-4,
        batch_size=16,
        seed=7,
        policy_repo_id="user/groot-run",
        wandb=True,
    )
    args = build_lerobot_args(req)
    assert f"--dataset.root={ds.resolve()}" in args
    assert "--policy.optimizer_lr=0.0001" in args
    assert "--batch_size=16" in args
    assert "--seed=7" in args
    assert "--policy.repo_id=user/groot-run" in args
    assert "--policy.push_to_hub=true" in args
    assert "--wandb.enable=true" in args
    assert "--policy.type=groot" in args
    assert "--policy.base_model_path=nvidia/GR00T-N1.7-3B" in args
    assert "--policy.embodiment_tag=new_embodiment" in args
    assert "--policy.tune_llm=false" in args
    assert "--policy.tune_visual=false" in args
    assert "--policy.tune_projector=true" in args
    assert "--policy.tune_diffusion_model=true" in args
    assert "--policy.use_relative_actions=true" in args
    assert '--policy.relative_exclude_joints=["hand"]' in args


def test_state_action_layout_is_26() -> None:
    assert len(STATE_ACTION_NAMES) == STATE_ACTION_DIM == 26
    assert STATE_ACTION_NAMES[0] == "left_arm_shoulder_pitch"
    assert STATE_ACTION_NAMES[7] == "right_arm_shoulder_pitch"
    assert STATE_ACTION_NAMES[14] == "left_hand_thumb_flex"
    assert STATE_ACTION_NAMES[15] == "left_hand_thumb_aux"
    assert STATE_ACTION_NAMES[20] == "right_hand_thumb_flex"
    assert STATE_ACTION_NAMES[-1] == "right_hand_pinky"


def test_parse_args_and_request(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--dataset",
            "team/ball_kick",
            "--gpu",
            "b100",
            "--policy",
            "molmoact2",
            "--steps",
            "1234",
        ]
    )
    req = request_from_args(args)
    assert req.dataset.repo_id == "team/ball_kick"
    assert req.gpu == "B200"
    assert req.policy == "molmoact2"
    assert req.steps == 1234
    assert req.lr is None
    assert req.batch_size is None
    assert req.output_dir.endswith("molmoact2_team_ball_kick")


def test_push_to_hub_requires_repo_id() -> None:
    args = parse_args(["--dataset", "user/ds", "--push-to-hub"])
    with pytest.raises(ValueError, match="policy-repo-id"):
        request_from_args(args)


def test_molmoact2_defaults_to_lora() -> None:
    args = parse_args(
        ["--dataset", "user/ds", "--policy", "molmoact2", "--steps", "1000"]
    )
    req = request_from_args(args)
    assert req.only_action_expert_ft is False
    lerobot_args = build_lerobot_args(req)
    assert "--policy.enable_lora_vlm=true" in lerobot_args
    assert all(not a.startswith("--policy.train_action_expert_only=") for a in lerobot_args)


def test_molmoact2_onlyactionexpertft() -> None:
    args = parse_args(
        [
            "--dataset",
            "user/ds",
            "--policy",
            "molmoact2",
            "--steps",
            "1000",
            "--onlyactionexpertft",
        ]
    )
    req = request_from_args(args)
    assert req.only_action_expert_ft is True
    lerobot_args = build_lerobot_args(req)
    assert "--policy.train_action_expert_only=true" in lerobot_args
    assert "--policy.enable_lora_vlm=false" in lerobot_args
    assert "--policy.action_mode=continuous" in lerobot_args


def test_onlyactionexpertft_requires_molmoact2_or_groot() -> None:
    args = parse_args(
        ["--dataset", "user/ds", "--policy", "act", "--onlyactionexpertft"]
    )
    with pytest.raises(ValueError, match="molmoact2 or groot"):
        request_from_args(args)


def test_groot_onlyactionexpertft_accepted() -> None:
    args = parse_args(
        [
            "--dataset",
            "user/ds",
            "--policy",
            "groot",
            "--steps",
            "1000",
            "--onlyactionexpertft",
        ]
    )
    req = request_from_args(args)
    assert req.policy == "groot"
    assert req.only_action_expert_ft is True
    lerobot_args = build_lerobot_args(req)
    assert "--policy.tune_llm=false" in lerobot_args


def test_groot_rejects_mismatched_local_dataset(tmp_path: Path) -> None:
    ds = tmp_path / "bad_ds"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text(
        '{"features": {"observation.state": {"shape": [6], "names": ["a"]}, '
        '"action": {"shape": [6], "names": ["a"]}}}'
    )
    args = parse_args(
        ["--dataset", str(ds), "--policy", "groot", "--steps", "100"]
    )
    with pytest.raises(ValueError, match="G1\\+Revo2"):
        request_from_args(args)
