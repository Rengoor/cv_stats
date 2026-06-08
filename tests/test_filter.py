"""
Tests for apply_filter in basics.py.

Covers:
  - Property 9: For any random uint8 frame, output has identical height and
    width, dtype uint8, and all values in [0, 255].
  - Smoke test: call apply_filter with a valid frame; verify it runs without
    exception.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from hypothesis import given, settings
from hypothesis.extra.numpy import arrays

from basics import apply_filter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Small shape keeps PBT fast while still exercising the filter
PBT_FRAME_SHAPE = (32, 32, 3)
FULL_FRAME_SHAPE = (720, 1280, 3)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Feature: cv-virtual-camera-pipeline, Property 9: For any valid uint8 frame,
# apply_filter returns a frame with identical height and width, dtype uint8,
# and all values in [0, 255].
@given(frame=arrays(dtype=np.uint8, shape=PBT_FRAME_SHAPE))
@settings(max_examples=100)
def test_property9_filter_preserves_dimensions_dtype_range(frame):
    """Property 9: filter preserves spatial dims, dtype uint8, and range [0,255]."""
    result = apply_filter(frame)

    assert result.shape[0] == frame.shape[0], (
        f"Height mismatch: expected {frame.shape[0]}, got {result.shape[0]}"
    )
    assert result.shape[1] == frame.shape[1], (
        f"Width mismatch: expected {frame.shape[1]}, got {result.shape[1]}"
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
# Smoke / example-based tests
# ---------------------------------------------------------------------------

class TestApplyFilterExamples:
    """Smoke and concrete example tests for apply_filter."""

    def test_smoke_valid_frame_no_exception(self):
        """Smoke test: apply_filter must not raise on a valid frame."""
        frame = np.random.randint(0, 256, FULL_FRAME_SHAPE, dtype=np.uint8)
        result = apply_filter(frame)  # must not raise
        assert result is not None

    def test_output_shape_matches_input(self):
        """Output shape must equal input shape exactly."""
        frame = np.random.randint(0, 256, FULL_FRAME_SHAPE, dtype=np.uint8)
        result = apply_filter(frame)
        assert result.shape == frame.shape, (
            f"Expected shape {frame.shape}, got {result.shape}"
        )

    def test_output_dtype_is_uint8(self):
        """Output dtype must be uint8."""
        frame = np.random.randint(0, 256, FULL_FRAME_SHAPE, dtype=np.uint8)
        result = apply_filter(frame)
        assert result.dtype == np.uint8

    def test_output_values_in_range(self):
        """All output pixels must be in [0, 255]."""
        frame = np.random.randint(0, 256, FULL_FRAME_SHAPE, dtype=np.uint8)
        result = apply_filter(frame)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_constant_frame_no_exception(self):
        """A constant-value frame must not raise and must return uint8."""
        frame = np.full(FULL_FRAME_SHAPE, 128, dtype=np.uint8)
        result = apply_filter(frame)
        assert result.dtype == np.uint8
        assert result.shape == FULL_FRAME_SHAPE

    def test_all_zeros_frame(self):
        """An all-zero frame must produce an all-zero output (blur of zeros is zeros)."""
        frame = np.zeros(FULL_FRAME_SHAPE, dtype=np.uint8)
        result = apply_filter(frame)
        np.testing.assert_array_equal(result, frame)

    def test_all_max_frame(self):
        """An all-255 frame: blur of constant is constant — output should be 255."""
        frame = np.full(FULL_FRAME_SHAPE, 255, dtype=np.uint8)
        result = apply_filter(frame)
        np.testing.assert_array_equal(result, frame)
