"""
Tests for linear_transform in basics.py.

Covers:
  - Property 5: For any random uint8 frame, linear_transform(frame, 1.0, 0.0)
    returns a frame whose pixel values are identical to the input.
  - Property 6: For any valid uint8 frame and finite alpha/beta, the output has
    the same shape, dtype uint8, and all values clamped to [0, 255].
  - Example-based unit test:
    - Channel independence: only R channel non-zero → G and B unchanged after transform.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from basics import linear_transform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FRAME_SHAPE = (720, 1280, 3)

# Strategies for finite, well-behaved alpha and beta values
alpha_strategy = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
beta_strategy = st.floats(min_value=-255.0, max_value=255.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Feature: cv-virtual-camera-pipeline, Property 5: For any valid uint8 frame,
# linear_transform(frame, 1.0, 0.0) returns identical pixel values to input.
@given(frame=arrays(dtype=np.uint8, shape=FRAME_SHAPE))
@settings(max_examples=100)
def test_property5_identity_transform(frame):
    """Property 5: identity transform (alpha=1.0, beta=0.0) returns pixel values equal to input."""
    result = linear_transform(frame, 1.0, 0.0)

    assert result.shape == frame.shape, \
        f"Shape mismatch: expected {frame.shape}, got {result.shape}"
    assert result.dtype == np.uint8, \
        f"dtype mismatch: expected uint8, got {result.dtype}"
    np.testing.assert_array_equal(result, frame,
        err_msg="Identity transform must return pixel values identical to input")


# Feature: cv-virtual-camera-pipeline, Property 6: For any valid uint8 frame and
# finite alpha/beta, output has same shape, dtype uint8, and all values in [0, 255].
@given(
    frame=arrays(dtype=np.uint8, shape=FRAME_SHAPE),
    alpha=alpha_strategy,
    beta=beta_strategy,
)
@settings(max_examples=100)
def test_property6_valid_output_shape_dtype_range(frame, alpha, beta):
    """Property 6: output has same shape, dtype uint8, and all values in [0, 255]."""
    result = linear_transform(frame, alpha, beta)

    assert result.shape == frame.shape, \
        f"Shape mismatch: expected {frame.shape}, got {result.shape}"
    assert result.dtype == np.uint8, \
        f"dtype mismatch: expected uint8, got {result.dtype}"
    assert result.min() >= 0, \
        f"Output contains values below 0: min={result.min()}"
    assert result.max() <= 255, \
        f"Output contains values above 255: max={result.max()}"


# ---------------------------------------------------------------------------
# Example-based unit tests
# ---------------------------------------------------------------------------

class TestLinearTransformExamples:
    """Concrete example tests for linear_transform."""

    def test_channel_independence_g_and_b_unchanged_when_r_only_nonzero(self):
        """
        Channel independence: frame with only R non-zero, G and B all zeros.
        After transform, G and B channels should remain all zeros (no channel mixing).
        """
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:, :, 0] = 100  # R = 100, G = 0, B = 0

        alpha = 2.0
        beta = 10.0
        result = linear_transform(frame, alpha, beta)

        # G channel was 0 → alpha * 0 + beta = beta = 10.0, clipped to 10
        expected_g = np.clip(alpha * 0 + beta, 0, 255).astype(np.uint8)
        expected_b = np.clip(alpha * 0 + beta, 0, 255).astype(np.uint8)

        np.testing.assert_array_equal(result[:, :, 1], expected_g,
            err_msg="G channel should be transformed independently from R")
        np.testing.assert_array_equal(result[:, :, 2], expected_b,
            err_msg="B channel should be transformed independently from R")

    def test_channel_independence_r_channel_unaffected_by_g_values(self):
        """
        Verify R channel values are not influenced by G or B channel values.
        R=50 with various G and B values should always produce the same R output.
        """
        frame1 = np.zeros((5, 5, 3), dtype=np.uint8)
        frame1[:, :, 0] = 50   # R=50, G=0, B=0

        frame2 = np.zeros((5, 5, 3), dtype=np.uint8)
        frame2[:, :, 0] = 50   # R=50
        frame2[:, :, 1] = 200  # G=200
        frame2[:, :, 2] = 100  # B=100

        alpha, beta = 1.5, 20.0
        result1 = linear_transform(frame1, alpha, beta)
        result2 = linear_transform(frame2, alpha, beta)

        np.testing.assert_array_equal(result1[:, :, 0], result2[:, :, 0],
            err_msg="R channel output must not depend on G or B values")

    def test_clamping_above_255(self):
        """Values that exceed 255 after transform should be clamped to 255."""
        frame = np.full((4, 4, 3), 200, dtype=np.uint8)
        result = linear_transform(frame, alpha=2.0, beta=100.0)

        # 2.0 * 200 + 100 = 500 → clamped to 255
        assert result.max() == 255, \
            f"Expected max=255 after clamping, got {result.max()}"
        np.testing.assert_array_equal(result, np.full((4, 4, 3), 255, dtype=np.uint8))

    def test_clamping_below_zero(self):
        """Values that fall below 0 after transform should be clamped to 0."""
        frame = np.full((4, 4, 3), 50, dtype=np.uint8)
        result = linear_transform(frame, alpha=1.0, beta=-100.0)

        # 1.0 * 50 - 100 = -50 → clamped to 0
        assert result.min() == 0, \
            f"Expected min=0 after clamping, got {result.min()}"
        np.testing.assert_array_equal(result, np.zeros((4, 4, 3), dtype=np.uint8))

    def test_returns_uint8_dtype(self):
        """Output dtype must always be uint8."""
        frame = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        result = linear_transform(frame, alpha=1.5, beta=30.0)
        assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"

    def test_output_shape_matches_input(self):
        """Output shape must match input shape exactly."""
        frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        result = linear_transform(frame, alpha=0.5, beta=50.0)
        assert result.shape == frame.shape, \
            f"Expected shape {frame.shape}, got {result.shape}"

    def test_alpha_zero_produces_constant_beta_frame(self):
        """alpha=0 collapses all pixels to clip(beta, 0, 255)."""
        frame = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        beta = 128.0
        result = linear_transform(frame, alpha=0.0, beta=beta)
        expected = np.full((8, 8, 3), int(beta), dtype=np.uint8)
        np.testing.assert_array_equal(result, expected,
            err_msg="alpha=0 should produce a constant frame equal to clip(beta,0,255)")
