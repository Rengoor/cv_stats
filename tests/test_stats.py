"""
Tests for compute_stats in basics.py.

Covers:
  - Property 1: For any random uint8 frame, mean/mode/max/min in [0,255] and std in [0,127.5]
  - Property 2: For any constant-fill frame, std == 0.0 for all channels
  - Example-based unit tests: known constant frames -> exact values for all five metrics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from basics import compute_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FRAME_SHAPE = (720, 1280, 3)


def make_frame(value_r: int, value_g: int, value_b: int) -> np.ndarray:
    """Create a solid-colour (H, W, 3) uint8 frame."""
    frame = np.empty(FRAME_SHAPE, dtype=np.uint8)
    frame[:, :, 0] = value_r
    frame[:, :, 1] = value_g
    frame[:, :, 2] = value_b
    return frame


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Feature: cv-virtual-camera-pipeline, Property 1: For any random uint8 frame,
# mean/mode/max/min in [0,255] and std in [0,127.5]
@given(frame=arrays(dtype=np.uint8, shape=FRAME_SHAPE))
@settings(max_examples=100)
def test_property1_stats_values_in_range(frame):
    """Property 1: statistical metrics are within their valid ranges for any uint8 frame."""
    stats = compute_stats(frame)

    for ch in range(3):
        assert 0.0 <= stats['mean'][ch] <= 255.0, \
            f"mean[{ch}]={stats['mean'][ch]} out of [0,255]"
        assert 0 <= stats['mode'][ch] <= 255, \
            f"mode[{ch}]={stats['mode'][ch]} out of [0,255]"
        assert 0.0 <= stats['std'][ch] <= 127.5, \
            f"std[{ch}]={stats['std'][ch]} out of [0,127.5]"
        assert 0 <= stats['max'][ch] <= 255, \
            f"max[{ch}]={stats['max'][ch]} out of [0,255]"
        assert 0 <= stats['min'][ch] <= 255, \
            f"min[{ch}]={stats['min'][ch]} out of [0,255]"


# Feature: cv-virtual-camera-pipeline, Property 2: For any constant-fill frame,
# std == 0.0 for all channels
@given(
    r=st.integers(min_value=0, max_value=255),
    g=st.integers(min_value=0, max_value=255),
    b=st.integers(min_value=0, max_value=255),
)
@settings(max_examples=100)
def test_property2_constant_frame_std_zero(r, g, b):
    """Property 2: a constant-fill frame must yield std == 0.0 for every channel."""
    frame = make_frame(r, g, b)
    stats = compute_stats(frame)

    for ch in range(3):
        assert stats['std'][ch] == 0.0, \
            f"std[{ch}]={stats['std'][ch]} expected 0.0 for constant frame"


# ---------------------------------------------------------------------------
# Example-based unit tests
# ---------------------------------------------------------------------------

class TestComputeStatsConstantFrames:
    """Verify exact metric values for known constant-colour frames."""

    def test_all_zeros_frame(self):
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        stats = compute_stats(frame)

        for ch in range(3):
            assert stats['mean'][ch] == 0.0
            assert stats['mode'][ch] == 0
            assert stats['std'][ch] == 0.0
            assert stats['max'][ch] == 0
            assert stats['min'][ch] == 0

    def test_all_255_frame(self):
        frame = np.full(FRAME_SHAPE, 255, dtype=np.uint8)
        stats = compute_stats(frame)

        for ch in range(3):
            assert stats['mean'][ch] == 255.0
            assert stats['mode'][ch] == 255
            assert stats['std'][ch] == 0.0
            assert stats['max'][ch] == 255
            assert stats['min'][ch] == 255

    def test_constant_128_frame(self):
        frame = np.full(FRAME_SHAPE, 128, dtype=np.uint8)
        stats = compute_stats(frame)

        for ch in range(3):
            assert stats['mean'][ch] == 128.0
            assert stats['mode'][ch] == 128
            assert stats['std'][ch] == 0.0
            assert stats['max'][ch] == 128
            assert stats['min'][ch] == 128

    def test_per_channel_constants(self):
        """Each channel holds a distinct constant value; check all metrics per channel."""
        frame = make_frame(10, 100, 200)
        stats = compute_stats(frame)

        expected = [10, 100, 200]
        for ch, val in enumerate(expected):
            assert stats['mean'][ch] == float(val), f"mean[{ch}]"
            assert stats['mode'][ch] == val, f"mode[{ch}]"
            assert stats['std'][ch] == 0.0, f"std[{ch}]"
            assert stats['max'][ch] == val, f"max[{ch}]"
            assert stats['min'][ch] == val, f"min[{ch}]"

    def test_return_dict_keys(self):
        """Result must contain exactly the required keys."""
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        stats = compute_stats(frame)
        assert set(stats.keys()) == {'mean', 'mode', 'std', 'max', 'min'}

    def test_return_values_are_3_tuples(self):
        """Each value in the dict must be a 3-element sequence (R, G, B)."""
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        stats = compute_stats(frame)
        for key in ('mean', 'mode', 'std', 'max', 'min'):
            assert len(stats[key]) == 3, f"{key} should have 3 elements"
