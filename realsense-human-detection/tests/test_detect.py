"""Unit tests for portable one-frame detection (no RealSense required)."""

from __future__ import annotations

import numpy as np
import pytest

from realsense_human_detection.depth import median_depth_m_from_array
from realsense_human_detection.guidance import angle_to_clock, direction_instruction


def test_angle_to_clock_ahead():
    assert angle_to_clock(0.0) == "12:00"


def test_angle_to_clock_right():
    assert angle_to_clock(30.0) == "1:00"


def test_direction_instruction_ahead():
    text = direction_instruction(0.0)
    assert "dead ahead" in text


def test_direction_instruction_move_left_when_person_on_right():
    text = direction_instruction(20.0)
    assert "move moderately left" in text


def test_median_depth_patch():
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[8:13, 8:13] = 1.5
    assert median_depth_m_from_array(depth, 10, 10, patch=2) == pytest.approx(1.5)


def test_median_depth_empty():
    depth = np.zeros((10, 10), dtype=np.float32)
    assert median_depth_m_from_array(depth, 5, 5) == 0.0


def test_human_detector_one_frame_smoke():
    """Loads YOLO and runs on a blank frame — must return a list (usually empty)."""
    pytest.importorskip("ultralytics")
    from realsense_human_detection import HumanDetector

    det = HumanDetector(device="cpu", conf_threshold=0.5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = det.detect(frame)
    assert isinstance(out, list)
    # Blank frame should not invent people.
    assert out == []


def test_human_detector_with_depth_fields():
    pytest.importorskip("ultralytics")
    from realsense_human_detection import HumanDetector, PersonDetection

    det = HumanDetector(device="cpu")
    color = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 0.8, dtype=np.float32)
    out = det.detect(color, depth_m=depth, fx=600.0, ppx=320.0)
    assert isinstance(out, list)
    for p in out:
        assert isinstance(p, PersonDetection)
        assert len(p.bbox) == 4
