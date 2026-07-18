"""Smoke-test Modal PolicyServer without a real G1 (no unitree/DDS).

Sends fake obs shaped like sudoping01/pi05_g1_boxmove_v2 and prints action
chunks so you can see joint commands change.

  uv run --package remote-vla-inference --extra client \\
    python remote-vla-inference/smoke_pi05_client.py \\
    --server_address=HOST:PORT
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time

import grpc
import numpy as np
import torch

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import hw_to_dataset_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_pi05")

DEFAULT_POLICY = "sudoping01/pi05_g1_boxmove_v2"


def make_lerobot_features() -> dict:
    obs_features = {f"motor_{i}.q": float for i in range(29)}
    obs_features["global_view"] = (480, 640, 3)
    return hw_to_dataset_features(obs_features, OBS_STR, use_video=False)


def fake_robot_obs(task: str, t: int) -> dict:
    state = 0.1 * np.sin(0.05 * t + np.arange(14) * 0.2)
    obs = {f"motor_{i}.q": 0.0 for i in range(29)}
    for i in range(14):
        obs[f"motor_{i}.q"] = float(state[i])
    obs["global_view"] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obs["task"] = task
    return obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server_address", required=True)
    p.add_argument("--policy", default=DEFAULT_POLICY)
    p.add_argument("--policy_type", default="pi05")
    p.add_argument("--task", default="move the blue box")
    p.add_argument("--policy_device", default="cuda")
    p.add_argument("--actions_per_chunk", type=int, default=50)
    p.add_argument("--rounds", type=int, default=5)
    args = p.parse_args()

    features = make_lerobot_features()
    policy_cfg = RemotePolicyConfig(
        args.policy_type,
        args.policy,
        features,
        args.actions_per_chunk,
        args.policy_device,
    )

    channel = grpc.insecure_channel(
        args.server_address, grpc_channel_options(initial_backoff="0.0333s")
    )
    stub = services_pb2_grpc.AsyncInferenceStub(channel)

    logger.info("Ready handshake → %s", args.server_address)
    stub.Ready(services_pb2.Empty())

    logger.info(
        "Sending policy setup: %s @ %s (HF download on Modal may take several minutes)",
        args.policy_type,
        args.policy,
    )
    stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(policy_cfg)))

    for r in range(args.rounds):
        timed = TimedObservation(
            timestamp=time.time(),
            timestep=r,
            observation=fake_robot_obs(args.task, r),
            must_go=True,
        )
        data_iterator = send_bytes_in_chunks(pickle.dumps(timed), services_pb2.Observation)
        stub.SendObservations(data_iterator)

        actions = None
        for _ in range(180):
            chunk = stub.GetActions(services_pb2.Empty())
            if len(chunk.data) > 0:
                actions = pickle.loads(chunk.data)
                break
            time.sleep(1.0)

        if not actions:
            logger.error("Round %d: no actions returned", r)
            continue

        first = actions[0].get_action().detach().cpu().reshape(-1)
        logger.info(
            "round=%d n_actions=%d action_dim=%d arm14=%s",
            r,
            len(actions),
            first.numel(),
            [round(float(x), 3) for x in first[:14].tolist()],
        )

    channel.close()
    logger.info("Smoke done.")


if __name__ == "__main__":
    main()
