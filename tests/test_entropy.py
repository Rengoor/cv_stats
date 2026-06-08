"""
Tests for compute_entropy in basics.py.

Covers:
  - Property 3: For any random uint8 frame, all returned entropies are floats
    in [0.0, 8.0] and two calls return identical results (determinism)
  - Example-based unit tests:
    - single-intensity channel -> 0.0
    - perfectly uniform channel -> 8.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from basics import compute_entropy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FRAME_SHAPE = (720, 1280, 3)


def make_uniform_channel_frame() -> np.ndarray:
    """
    Create a frame where each channel has a perfectly uniform distribution
    across all 256 intensity values (each value appears equally often).
    Total pixels per channel = 720 * 1280 = 921600.
    921600 / 256 = 3600 pixels per intensity level -> perfectly uniform.
    """
    pixels_per_channel = 720 * 1280  # 921600
    values_per_level = pixels_per_channel // 256  # 3600

    channel = np.repeat(np.arange(256, dtype=np.uint8), values_per_level)
    # channel has exactly 921600 elements
    frame = np.stack([channel, channel, channel], axis=-1).reshape(FRAME_SHAPE)
    return frame


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Feature: cv-virtual-camera-pipeline, Property 3: For any random uint8 frame,
# all returned entropies are floats in [0.0, 8.0] and two calls return identical
# results (determinism)
@given(frame=arrays(dtype=np.uint8, shape=FRAME_SHAPE))
@settings(max_examples=100)
def test_property3_entropy_range_and_determinism(frame):
    """Property 3: entropy values are in [0.0, 8.0] and the function is deterministic."""
    result1 = compute_entropy(frame)
    result2 = compute_entropy(frame)

    assert len(result1) == 3, "Expected 3-tuple return value"
    assert len(result2) == 3, "Expected 3-tuple return value on second call"

    for ch in range(3):
        assert isinstance(result1[ch], float), \
            f"entropy[{ch}] should be a float, got {type(result1[ch])}"
        assert 0.0 <= result1[ch] <= 8.0, \
            f"entropy[{ch}]={result1[ch]} out of [0.0, 8.0]"
        assert result1[ch] == result2[ch], \
            f"entropy[{ch}] not deterministic: {result1[ch]} != {result2[ch]}"


# ---------------------------------------------------------------------------
# Example-based unit tests
# ---------------------------------------------------------------------------

class TestComputeEntropyExamples:
    """Concrete example tests for edge cases of compute_entropy."""

    def test_single_intensity_channel_returns_zero_entropy(self):
        """A channel with all pixels at the same intensity has entropy 0.0."""
        frame = np.full(FRAME_SHAPE, 128, dtype=np.uint8)
        ent_r, ent_g, ent_b = compute_entropy(frame)

        assert ent_r == pytest.approx(0.0, abs=1e-10), \
            f"Expected entropy_r=0.0 for constant channel, got {ent_r}"
        assert ent_g == pytest.approx(0.0, abs=1e-10), \
            f"Expected entropy_g=0.0 for constant channel, got {ent_g}"
        assert ent_b == pytest.approx(0.0, abs=1e-10), \
            f"Expected entropy_b=0.0 for constant channel, got {ent_b}"

    def test_single_intensity_zero_channel_returns_zero_entropy(self):
        """Edge case: all pixels = 0."""
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        ent_r, ent_g, ent_b = compute_entropy(frame)

        assert ent_r == pytest.approx(0.0, abs=1e-10)
        assert ent_g == pytest.approx(0.0, abs=1e-10)
        assert ent_b == pytest.approx(0.0, abs=1e-10)

    def test_single_intensity_255_channel_returns_zero_entropy(self):
        """Edge case: all pixels = 255."""
        frame = np.full(FRAME_SHAPE, 255, dtype=np.uint8)
        ent_r, ent_g, ent_b = compute_entropy(frame)

        assert ent_r == pytest.approx(0.0, abs=1e-10)
        assert ent_g == pytest.approx(0.0, abs=1e-10)
        assert ent_b == pytest.approx(0.0, abs=1e-10)

    def test_uniform_distribution_returns_8_bits_entropy(self):
        """A perfectly uniform channel distribution yields entropy = 8.0 bits."""
        frame = make_uniform_channel_frame()
        ent_r, ent_g, ent_b = compute_entropy(frame)

        assert ent_r == pytest.approx(8.0, abs=1e-10), \
            f"Expected entropy_r=8.0 for uniform channel, got {ent_r}"
        assert ent_g == pytest.approx(8.0, abs=1e-10), \
            f"Expected entropy_g=8.0 for uniform channel, got {ent_g}"
        assert ent_b == pytest.approx(8.0, abs=1e-10), \
            f"Expected entropy_b=8.0 for uniform channel, got {ent_b}"

    def test_returns_three_floats(self):
        """Return value must be a 3-tuple of floats."""
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        result = compute_entropy(frame)

        assert len(result) == 3, "Expected 3-tuple"
        for ch, val in enumerate(result):
            assert isinstance(val, float), \
                f"result[{ch}]={val!r} is {type(val)}, expected float"

    def test_per_channel_independence(self):
        """Different constant values per channel should all yield entropy 0.0."""
        frame = np.empty(FRAME_SHAPE, dtype=np.uint8)
        frame[:, :, 0] = 10   # R
        frame[:, :, 1] = 100  # G
        frame[:, :, 2] = 200  # B
        ent_r, ent_g, ent_b = compute_entropy(frame)

        assert ent_r == pytest.approx(0.0, abs=1e-10)
        assert ent_g == pytest.approx(0.0, abs=1e-10)
        assert ent_b == pytest.approx(0.0, abs=1e-10)
