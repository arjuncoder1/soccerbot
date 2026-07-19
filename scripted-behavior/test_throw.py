"""Unit tests for the gentle goalkeeper push (no hardware/sim connection).

throw.py no longer drives to a fixed absolute pose -- it applies a relative
delta on top of whatever the current arm position is. These tests check that
delta against a family of plausible starting poses (g1_arm_fk.py, sourced
from the real URDF), not just internal consistency -- the point is to catch
a delta that only happens to work from one lucky starting point.
"""

from __future__ import annotations

from g1_arm_fk import LEFT_ARM, check_limits, left_elbow_position, left_hand_position
from throw import (
    ARM_JOINT_KEYS,
    ARM_JOINTS,
    _CANDIDATE_HOLDING_POSES,
    _FK_NAME_LEFT,
    _mirror_for_check,
    _push_target,
)


def test_arm_joints_is_14() -> None:
    assert len(ARM_JOINTS) == 14
    assert len(set(ARM_JOINTS)) == 14  # no duplicates
    assert all(key == f"{joint}.q" for joint, key in zip(ARM_JOINTS, ARM_JOINT_KEYS))


def test_push_target_covers_all_arm_joints() -> None:
    for candidate in _CANDIDATE_HOLDING_POSES:
        start_pose = _mirror_for_check(candidate)
        release_pose = _push_target(start_pose)
        assert set(release_pose.keys()) == set(ARM_JOINT_KEYS)


def test_push_never_touches_legs_or_waist() -> None:
    leg_waist_markers = ("Hip", "Knee", "Ankle", "Waist")
    for candidate in _CANDIDATE_HOLDING_POSES:
        release_pose = _push_target(_mirror_for_check(candidate))
        for joint in release_pose:
            assert not any(marker in joint for marker in leg_waist_markers)


def test_shoulder_roll_and_yaw_are_preserved() -> None:
    """The whole point of the relative-delta design: whatever left/right hand
    placement the current hold already has must be preserved, not overwritten
    with a fixed absolute value."""
    for candidate in _CANDIDATE_HOLDING_POSES:
        start_pose = _mirror_for_check(candidate)
        release_pose = _push_target(start_pose)
        assert release_pose["kLeftShoulderRoll.q"] == start_pose["kLeftShoulderRoll.q"]
        assert release_pose["kLeftShoulderYaw.q"] == start_pose["kLeftShoulderYaw.q"]
        assert release_pose["kRightShoulderRoll.q"] == start_pose["kRightShoulderRoll.q"]
        assert release_pose["kRightShoulderYaw.q"] == start_pose["kRightShoulderYaw.q"]


def test_within_real_g1_joint_limits_for_every_candidate() -> None:
    """Every candidate starting pose, pushed forward, must stay inside the
    real G1 29-DoF URDF limits (clamping guarantees this, but confirm)."""
    for candidate in _CANDIDATE_HOLDING_POSES:
        release_pose = _push_target(_mirror_for_check(candidate))
        left_fk = {fk: release_pose[key] for key, fk in _FK_NAME_LEFT.items()}
        assert check_limits(LEFT_ARM, left_fk) == [], candidate


def test_push_moves_hand_forward_for_every_candidate() -> None:
    """The core requirement, checked against every candidate starting pose,
    not just one: the push must move the hand forward by a real amount."""
    for candidate in _CANDIDATE_HOLDING_POSES:
        start_hand = left_hand_position(candidate)
        release_pose = _push_target(_mirror_for_check(candidate))
        left_fk = {fk: release_pose[key] for key, fk in _FK_NAME_LEFT.items()}
        release_hand = left_hand_position(left_fk)
        gain = release_hand[0] - start_hand[0]
        assert gain > 0.05, f"candidate {candidate}: only {gain:.3f}m forward"


def test_elbow_never_swings_behind_torso_for_every_candidate() -> None:
    """The elbow joint itself (not just the hand) must stay in front of the
    torso for every candidate. A hand path can look fine numerically while
    the elbow bends backward to get there -- this was a real bug caught by
    actually rendering the motion, not by hand-position checks alone."""
    for candidate in _CANDIDATE_HOLDING_POSES:
        release_pose = _push_target(_mirror_for_check(candidate))
        left_fk = {fk: release_pose[key] for key, fk in _FK_NAME_LEFT.items()}
        elbow_pos = left_elbow_position(left_fk)
        assert elbow_pos[0] >= 0.0, f"candidate {candidate}: elbow behind torso, x={elbow_pos[0]:.3f}"


def test_hands_never_cross_centerline_for_every_candidate() -> None:
    """Two hands holding one ball approach it from either side -- the left
    hand must never cross to the body's right side (y<0) after the push, for
    any candidate starting pose. An earlier version of this trajectory let
    the hands cross, which renders as crossed arms rather than two hands
    cupping a ball."""
    for candidate in _CANDIDATE_HOLDING_POSES:
        release_pose = _push_target(_mirror_for_check(candidate))
        left_fk = {fk: release_pose[key] for key, fk in _FK_NAME_LEFT.items()}
        release_hand = left_hand_position(left_fk)
        assert release_hand[1] >= 0.0, f"candidate {candidate}: hand crosses centerline, y={release_hand[1]:.3f}"


def test_candidates_are_themselves_valid_holding_poses() -> None:
    """Sanity-check the test fixtures: every candidate starting pose must
    already be a plausible "ball held in front of the torso, own side" pose
    (hand in front of the body, on its own side) -- otherwise a passing test
    doesn't actually mean anything. Caught a bad fixture once already (a
    candidate whose hand started already crossed to the wrong side)."""
    for candidate in _CANDIDATE_HOLDING_POSES:
        hand = left_hand_position(candidate)
        assert hand[0] > 0.0, f"candidate {candidate}: hand not in front of torso, x={hand[0]:.3f}"
        assert hand[1] >= 0.0, f"candidate {candidate}: hand already crossed centerline, y={hand[1]:.3f}"
