"""
Tests for histogram_equalization in basics.py.

Covers:
  - Property 7: For any random uint8 frame, applying histogram_equalization twice
    returns identical pixel values on the second call as the first (idempotent).
  - Property 8: For any random uint8 frame, output has same shape, dtype uint8,
    and all values in [0, 255].
  - Example-based unit test: equalization matches cv2.equalizeHist applied per channel.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
import cv2

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from basics import histogram_equalization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FRAME_SHAPE = (720, 1280, 3)

# Smaller shape used for property tests to keep runtime manageable
PBT_FRAME_SHAPE = (32, 32, 3)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Feature: cv-virtual-camera-pipeline, Property 7: For any random uint8 frame,
# applying histogram_equalization twice yields identical pixel values on second call.
@given(frame=arrays(dtype=np.uint8, shape=PBT_FRAME_SHAPE))
@settings(max_examples=100)
def test_property7_equalization_is_idempotent(frame):
    """Property 7: histogram_equalization is idempotent — f(f(x)) == f(x) for any frame."""
    first = histogram_equalization(frame)
    second = histogram_equalization(first)

    np.testing.assert_array_equal(
        second,
        first,
        err_msg="Applying histogram_equalization twice must yield identical results",
    )


# Feature: cv-virtual-camera-pipeline, Property 8: For any random uint8 frame,
# output has same shape, dtype uint8, and all values in [0, 255].
@given(frame=arrays(dtype=np.uint8, shape=PBT_FRAME_SHAPE))
@settings(max_examples=100)
def test_property8_valid_output_shape_dtype_range(frame):
    """Property 8: output has same shape, dtype uint8, and all values in [0, 255]."""
    result = histogram_equalization(frame)

    assert result.shape == frame.shape, (
        f"Shape mismatch: expected {frame.shape}, got {result.shape}"
    )
    assert result.dtype == np.uint8, (
        f"dtype mismatch: expected uint8, got {result.dtype}"
    )
    assert int(result.min()) >= 0, (
        f"Output contains values below 0: min={result.min()}"
    )
    assert int(result.max()) <= 255, (
        f"Output contains values above 255: max={result.max()}"
    )


# ---------------------------------------------------------------------------
# Example-based unit tests
# ---------------------------------------------------------------------------

class TestHistogramEqualizationExamples:
    """Concrete example tests for histogram_equalization."""

    def _cv2_equalize_per_channel(self, frame: np.ndarray) -> np.ndarray:
        """Reference implementation: apply cv2.equalizeHist to each channel."""
        result = np.empty_like(frame)
        for ch in range(3):
            result[:, :, ch] = cv2.equalizeHist(frame[:, :, ch])
        return result

    def test_matches_cv2_equalize_hist_random_frame(self):
        """Equalization must match cv2.equalizeHist applied independently per channel."""
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, FRAME_SHAPE, dtype=np.uint8)

        result = histogram_equalization(frame)
        expected = self._cv2_equalize_per_channel(frame)

        np.testing.assert_array_equal(
            result,
            expected,
            err_msg="histogram_equalization output must match cv2.equalizeHist per channel",
        )

    def test_matches_cv2_equalize_hist_gradient_frame(self):
        """Test with a frame that has a gradient pattern."""
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        # Fill each channel with a ramp 0..255 tiled across width
        for ch in range(3):
            row = np.linspace(0, 255, FRAME_SHAPE[1], dtype=np.uint8)
            frame[:, :, ch] = np.tile(row, (FRAME_SHAPE[0], 1))

        result = histogram_equalization(frame)
        expected = self._cv2_equalize_per_channel(frame)

        np.testing.assert_array_equal(result, expected)

    def test_output_shape_matches_input(self):
        """Output shape must equal input shape."""
        frame = np.random.randint(0, 256, FRAME_SHAPE, dtype=np.uint8)
        result = histogram_equalization(frame)
        assert result.shape == frame.shape

    def test_output_dtype_is_uint8(self):
        """Output dtype must always be uint8."""
        frame = np.random.randint(0, 256, FRAME_SHAPE, dtype=np.uint8)
        result = histogram_equalization(frame)
        assert result.dtype == np.uint8

    def test_constant_frame_does_not_crash(self):
        """A constant-value frame (edge case) must not raise an exception."""
        frame = np.full(FRAME_SHAPE, 128, dtype=np.uint8)
        result = histogram_equalization(frame)
        assert result.dtype == np.uint8
        assert result.shape == FRAME_SHAPE

    def test_all_zeros_frame(self):
        """An all-zero frame should produce an all-zero output."""
        frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        result = histogram_equalization(frame)
        expected = self._cv2_equalize_per_channel(frame)
        np.testing.assert_array_equal(result, expected)

    def test_all_max_frame(self):
        """An all-255 frame should produce an all-255 output (or match cv2)."""
        frame = np.full(FRAME_SHAPE, 255, dtype=np.uint8)
        result = histogram_equalization(frame)
        expected = self._cv2_equalize_per_channel(frame)
        np.testing.assert_array_equal(result, expected)

    def test_idempotent_on_concrete_frame(self):
        """Concrete idempotency check: f(f(x)) == f(x)."""
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 256, FRAME_SHAPE, dtype=np.uint8)

        first = histogram_equalization(frame)
        second = histogram_equalization(first)

        np.testing.assert_array_equal(
            second,
            first,
            err_msg="histogram_equalization must be idempotent",
        )
