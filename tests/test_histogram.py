"""
Tests for histogram_figure_numba in basics.py.

Covers:
  - Property 4: For any random uint8 frame, all three returned arrays have
    length 256 and all elements are non-negative.
  - Example-based unit tests:
    - Frame with all Red pixels == 128 -> only r_bars[128] is non-zero
    - Verify @njit decorator is present on histogram_figure_numba
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from basics import histogram_figure_numba


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FRAME_SHAPE = (16, 16, 3)   # small frame for PBT speed (JIT overhead is O(1))
FULL_FRAME_SHAPE = (720, 1280, 3)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Feature: cv-virtual-camera-pipeline, Property 4: For any random uint8 frame,
# histogram_figure_numba returns exactly three arrays each of length 256 where
# every element is a non-negative integer.
@given(frame=arrays(dtype=np.uint8, shape=FRAME_SHAPE))
@settings(max_examples=100, deadline=None)
def test_property4_histogram_length_256_and_nonnegative(frame):
    """Property 4: all three bars arrays have length 256 and non-negative counts."""
    r_bars, g_bars, b_bars = histogram_figure_numba(frame)

    # Length check
    assert len(r_bars) == 256, f"r_bars has length {len(r_bars)}, expected 256"
    assert len(g_bars) == 256, f"g_bars has length {len(g_bars)}, expected 256"
    assert len(b_bars) == 256, f"b_bars has length {len(b_bars)}, expected 256"

    # Non-negative check
    for i in range(256):
        assert r_bars[i] >= 0, f"r_bars[{i}]={r_bars[i]} is negative"
        assert g_bars[i] >= 0, f"g_bars[{i}]={g_bars[i]} is negative"
        assert b_bars[i] >= 0, f"b_bars[{i}]={b_bars[i]} is negative"


# ---------------------------------------------------------------------------
# Example-based unit tests
# ---------------------------------------------------------------------------

class TestHistogramFigureNumbaExamples:
    """Concrete example tests for histogram_figure_numba."""

    def test_all_red_pixels_128_only_r_bars_128_nonzero(self):
        """Frame with all Red pixels == 128: only r_bars[128] is non-zero."""
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[:, :, 0] = 128   # R channel = 128
        # G and B channels remain 0

        r_bars, g_bars, b_bars = histogram_figure_numba(frame)

        # r_bars: only index 128 should be non-zero
        assert r_bars[128] > 0, "r_bars[128] should be non-zero"
        for i in range(256):
            if i != 128:
                assert r_bars[i] == 0, (
                    f"r_bars[{i}]={r_bars[i]} should be 0 when all Red pixels are 128"
                )

        # g_bars and b_bars: only index 0 should be non-zero
        assert g_bars[0] > 0, "g_bars[0] should count the zero-valued green pixels"
        for i in range(1, 256):
            assert g_bars[i] == 0, f"g_bars[{i}] should be 0"

        assert b_bars[0] > 0, "b_bars[0] should count the zero-valued blue pixels"
        for i in range(1, 256):
            assert b_bars[i] == 0, f"b_bars[{i}] should be 0"

    def test_r_bars_128_count_equals_pixel_count(self):
        """r_bars[128] should equal total number of pixels (H * W)."""
        h, w = 4, 8
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = 128

        r_bars, _, _ = histogram_figure_numba(frame)

        assert r_bars[128] == h * w, (
            f"r_bars[128]={r_bars[128]}, expected {h * w}"
        )

    def test_njit_decorator_present(self):
        """Verify @njit decorator is present on histogram_figure_numba."""
        # numba.core.registry.CPUDispatcher is the type of an @njit function.
        # We check this without importing numba internals directly.
        try:
            from numba.core.registry import CPUDispatcher
            assert isinstance(histogram_figure_numba, CPUDispatcher), (
                "histogram_figure_numba should be decorated with @njit "
                f"(got type {type(histogram_figure_numba)})"
            )
        except ImportError:
            # Older numba versions
            type_name = type(histogram_figure_numba).__name__
            assert "Dispatcher" in type_name, (
                "histogram_figure_numba should be decorated with @njit "
                f"(got type {type_name})"
            )

    def test_total_counts_equal_pixel_count(self):
        """Sum of all bars should equal total pixels in the frame."""
        h, w = 6, 10
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)

        r_bars, g_bars, b_bars = histogram_figure_numba(frame)
        total = h * w

        assert sum(r_bars) == total, f"sum(r_bars)={sum(r_bars)}, expected {total}"
        assert sum(g_bars) == total, f"sum(g_bars)={sum(g_bars)}, expected {total}"
        assert sum(b_bars) == total, f"sum(b_bars)={sum(b_bars)}, expected {total}"

    def test_returns_three_length_256_arrays(self):
        """Return value is a 3-tuple of length-256 arrays."""
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        result = histogram_figure_numba(frame)

        assert len(result) == 3, f"Expected 3-tuple, got {len(result)} items"
        r_bars, g_bars, b_bars = result
        assert len(r_bars) == 256
        assert len(g_bars) == 256
        assert len(b_bars) == 256
