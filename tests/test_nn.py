"""
Tests for nn_inference in basics.py.

Covers:
  - Property 10: For any valid uint8 frame, output has same height/width,
    dtype uint8, and all values in [0, 255].
  - Example-based unit tests:
    - Solid-colour frame (no detectable face) → output equals input.
    - Model is accessible (no RuntimeError on import).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from hypothesis import given, settings
from hypothesis.extra.numpy import arrays

from basics import nn_inference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Small shape keeps PBT fast; nn_inference scales with content, not resolution.
PBT_FRAME_SHAPE = (64, 64, 3)
FULL_FRAME_SHAPE = (720, 1280, 3)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Feature: cv-virtual-camera-pipeline, Property 10: For any valid uint8 frame,
# nn_inference returns a frame with same height/width, dtype uint8, and all
# values in [0, 255].
@given(frame=arrays(dtype=np.uint8, shape=PBT_FRAME_SHAPE))
@settings(max_examples=50, deadline=None)
def test_property10_nn_inference_valid_output(frame):
    """Property 10: output has same spatial dims, dtype uint8, values in [0,255]."""
    result = nn_inference(frame)

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
# Example-based unit tests
# ---------------------------------------------------------------------------

class TestNNInferenceExamples:
    """Concrete example tests for nn_inference."""

    def test_model_loads_without_error(self):
        """The MediaPipe model must be loadable at import time (no RuntimeError)."""
        # If the import succeeded, the model loaded. A simple call confirms it.
        frame = np.zeros(FULL_FRAME_SHAPE, dtype=np.uint8)
        result = nn_inference(frame)  # must not raise
        assert result is not None

    def test_solid_colour_frame_returns_input_unchanged(self):
        """A solid-colour frame contains no detectable face — output must equal input."""
        frame = np.full(FULL_FRAME_SHAPE, 128, dtype=np.uint8)
        result = nn_inference(frame)
        np.testing.assert_array_equal(
            result, frame,
            err_msg="Solid-colour frame should be returned unmodified (no face detected)",
        )

    def test_all_zeros_frame_returns_input(self):
        """An all-zero frame has no detectable face — must return original."""
        frame = np.zeros(FULL_FRAME_SHAPE, dtype=np.uint8)
        result = nn_inference(frame)
        np.testing.assert_array_equal(result, frame)

    def test_all_max_frame_returns_input(self):
        """An all-255 frame has no detectable face — must return original."""
        frame = np.full(FULL_FRAME_SHAPE, 255, dtype=np.uint8)
        result = nn_inference(frame)
        np.testing.assert_array_equal(result, frame)

    def test_output_shape_matches_input(self):
        """Output shape must always match input shape."""
        frame = np.random.randint(0, 256, FULL_FRAME_SHAPE, dtype=np.uint8)
        result = nn_inference(frame)
        assert result.shape == frame.shape

    def test_output_dtype_is_uint8(self):
        """Output dtype must always be uint8."""
        frame = np.random.randint(0, 256, FULL_FRAME_SHAPE, dtype=np.uint8)
        result = nn_inference(frame)
        assert result.dtype == np.uint8
