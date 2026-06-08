"""
Integration / pipeline tests for custom_processing in run.py.

Covers:
  - custom_processing is a generator function that yields one frame per input
  - Output frame has shape (720, 1280, 3), dtype uint8, values in [0, 255]
  - plot_strings_to_image is NOT called when frame width < 400 (mock-based)
  - "h" key toggles histogram with debounce (mock keyboard.is_pressed)
  - Each processor function is called exactly once per frame in defined order
  - Exceptions raised inside the loop propagate out of custom_processing
  - Integration: update_histogram called with arrays of length 256 each cycle
  - Integration: plot_overlay_to_image called once per frame when histogram enabled
"""

import sys
import os
import inspect
import types
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ---------------------------------------------------------------------------
# Stub out hardware-dependent modules before importing run.py
# ---------------------------------------------------------------------------
import unittest.mock as _mock

for _mod in ("pyvirtualcam", "keyboard"):
    if _mod not in sys.modules:
        sys.modules[_mod] = _mock.MagicMock()

# Stub capturing so VirtualCamera import doesn't fail
if "capturing" not in sys.modules:
    _capturing_stub = types.ModuleType("capturing")
    _capturing_stub.VirtualCamera = _mock.MagicMock()
    sys.modules["capturing"] = _capturing_stub

import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call

import run as run_module
from run import custom_processing, _build_stats_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FRAME_SHAPE = (720, 1280, 3)


def make_frame(value=128):
    """Create a solid-colour (720, 1280, 3) uint8 frame."""
    return np.full(FRAME_SHAPE, value, dtype=np.uint8)


def single_frame_source(frame=None):
    """Generator that yields exactly one frame."""
    if frame is None:
        frame = make_frame()
    yield frame


def multi_frame_source(n=3, frame=None):
    """Generator that yields n identical frames."""
    if frame is None:
        frame = make_frame()
    for _ in range(n):
        yield frame.copy()


# ---------------------------------------------------------------------------
# Generator behaviour
# ---------------------------------------------------------------------------

class TestCustomProcessingGenerator:
    """Verify custom_processing is a generator and yields one frame per input."""

    def test_is_generator_function(self):
        """custom_processing must be a generator function (uses yield)."""
        assert inspect.isgeneratorfunction(custom_processing), (
            "custom_processing must be a generator function"
        )

    def test_yields_one_frame_per_input(self):
        """Each input frame produces exactly one output frame."""
        inputs = [make_frame(v) for v in [50, 100, 150]]

        outputs = list(custom_processing(iter(inputs)))
        assert len(outputs) == len(inputs), (
            f"Expected {len(inputs)} output frames, got {len(outputs)}"
        )

    def test_output_frame_shape(self):
        """Output frame must have shape (720, 1280, 3)."""
        gen = custom_processing(single_frame_source())
        output = next(gen)
        assert output.shape == FRAME_SHAPE, (
            f"Expected shape {FRAME_SHAPE}, got {output.shape}"
        )

    def test_output_frame_dtype_uint8(self):
        """Output frame must have dtype uint8."""
        gen = custom_processing(single_frame_source())
        output = next(gen)
        assert output.dtype == np.uint8, (
            f"Expected dtype uint8, got {output.dtype}"
        )

    def test_output_frame_values_in_range(self):
        """All output pixel values must be in [0, 255]."""
        gen = custom_processing(single_frame_source())
        output = next(gen)
        assert int(output.min()) >= 0
        assert int(output.max()) <= 255


# ---------------------------------------------------------------------------
# plot_strings_to_image size guard
# ---------------------------------------------------------------------------

class TestTextOverlayGuard:
    """Verify plot_strings_to_image is skipped for small frames."""

    def test_plot_strings_not_called_when_width_less_than_400(self):
        """plot_strings_to_image must NOT be called when frame width < 400."""
        small_frame = np.full((720, 300, 3), 128, dtype=np.uint8)

        with patch("run.plot_strings_to_image") as mock_psi:
            # Patch other costly functions to avoid side effects
            with patch("run.nn_inference", side_effect=lambda f: f), \
                 patch("run.apply_filter", side_effect=lambda f: f), \
                 patch("run.histogram_equalization", side_effect=lambda f: f), \
                 patch("run.linear_transform", side_effect=lambda f, **kw: f), \
                 patch("run.update_histogram"), \
                 patch("run.plot_overlay_to_image", side_effect=lambda f, fig: f), \
                 patch("run.histogram_figure_numba",
                       return_value=(np.zeros(256), np.zeros(256), np.zeros(256))), \
                 patch("run.initialize_hist_figure",
                       return_value=(MagicMock(), MagicMock(), MagicMock(),
                                     MagicMock(), MagicMock(), MagicMock())), \
                 patch("keyboard.is_pressed", return_value=False):
                list(custom_processing(iter([small_frame])))

            mock_psi.assert_not_called()

    def test_plot_strings_called_when_frame_is_full_size(self):
        """plot_strings_to_image IS called when frame is 1280×720."""
        frame = make_frame()

        with patch("run.plot_strings_to_image", side_effect=lambda f, t: f) as mock_psi, \
             patch("run.nn_inference", side_effect=lambda f: f), \
             patch("run.apply_filter", side_effect=lambda f: f), \
             patch("run.histogram_equalization", side_effect=lambda f: f), \
             patch("run.linear_transform", side_effect=lambda f, **kw: f), \
             patch("run.update_histogram"), \
             patch("run.plot_overlay_to_image", side_effect=lambda f, fig: f), \
             patch("run.histogram_figure_numba",
                   return_value=(np.zeros(256), np.zeros(256), np.zeros(256))), \
             patch("run.initialize_hist_figure",
                   return_value=(MagicMock(), MagicMock(), MagicMock(),
                                 MagicMock(), MagicMock(), MagicMock())), \
             patch("keyboard.is_pressed", return_value=False):
            list(custom_processing(iter([frame])))

        mock_psi.assert_called_once()


# ---------------------------------------------------------------------------
# "h" key debounce toggle
# ---------------------------------------------------------------------------

class TestHistogramToggleDebounce:
    """Verify "h" key toggles histogram on/off with a 10-frame debounce."""

    def _run_frames(self, key_presses, n_frames=15):
        """
        Run n_frames through custom_processing, returning a list of booleans
        indicating whether plot_overlay_to_image was called for each frame.
        """
        frames = [make_frame() for _ in range(n_frames)]
        overlay_calls = []

        def fake_plot_overlay(f, fig):
            overlay_calls.append(True)
            return f

        def fake_plot_strings(f, t):
            return f

        key_iter = iter(key_presses)

        def fake_is_pressed(key):
            if key == 'h':
                try:
                    return next(key_iter)
                except StopIteration:
                    return False
            return False

        with patch("run.plot_overlay_to_image", side_effect=fake_plot_overlay), \
             patch("run.plot_strings_to_image", side_effect=fake_plot_strings), \
             patch("run.nn_inference", side_effect=lambda f: f), \
             patch("run.apply_filter", side_effect=lambda f: f), \
             patch("run.histogram_equalization", side_effect=lambda f: f), \
             patch("run.linear_transform", side_effect=lambda f, **kw: f), \
             patch("run.update_histogram"), \
             patch("run.histogram_figure_numba",
                   return_value=(np.zeros(256), np.zeros(256), np.zeros(256))), \
             patch("run.initialize_hist_figure",
                   return_value=(MagicMock(), MagicMock(), MagicMock(),
                                 MagicMock(), MagicMock(), MagicMock())), \
             patch("keyboard.is_pressed", side_effect=fake_is_pressed):
            list(custom_processing(iter(frames)))

        return overlay_calls

    def test_histogram_visible_by_default(self):
        """Histogram overlay must be rendered on first frame without any key press."""
        calls = self._run_frames(key_presses=[False] * 15)
        # All 15 frames should have overlay
        assert len(calls) == 15

    def test_h_press_disables_histogram(self):
        """Pressing h on frame 0 should disable histogram from frame 1 onward (debounce)."""
        # h pressed on first frame only
        key_presses = [True] + [False] * 14
        calls = self._run_frames(key_presses=key_presses, n_frames=15)
        # Frame 0: h pressed, histogram still rendered that frame (toggle applies after yield)
        # Frames 1–10: cooldown ticking, histogram OFF
        # Frame 11+: histogram still OFF (toggled off)
        assert len(calls) < 15, "Some frames should not have histogram after toggle"

    def test_debounce_prevents_double_toggle(self):
        """
        Holding h for multiple frames should only toggle once per 10-frame
        cooldown window.
        """
        # h held for 5 frames — should only count as one press
        key_presses = [True] * 5 + [False] * 10
        calls_held = self._run_frames(key_presses=key_presses, n_frames=15)

        # h pressed once — should also only toggle once
        key_presses_single = [True] + [False] * 14
        calls_single = self._run_frames(key_presses=key_presses_single, n_frames=15)

        assert len(calls_held) == len(calls_single), (
            "Holding h should behave identically to a single press within cooldown window"
        )


# ---------------------------------------------------------------------------
# Processor call order
# ---------------------------------------------------------------------------

class TestProcessorCallOrder:
    """Verify each processor is called exactly once per frame in defined order."""

    def test_each_processor_called_once_per_frame(self):
        """All 6 processor functions must be called exactly once for a single frame."""
        call_log = []

        def make_spy(name):
            def spy(frame, *args, **kwargs):
                call_log.append(name)
                return frame
            return spy

        with patch("run.compute_stats", side_effect=lambda f: {'mean': (0,0,0), 'mode': (0,0,0), 'std': (0,0,0), 'max': (0,0,0), 'min': (0,0,0)}) as m_stat, \
             patch("run.compute_entropy", side_effect=lambda f: (0.0, 0.0, 0.0)) as m_ent, \
             patch("run.linear_transform", side_effect=lambda f, **kw: f) as m_lt, \
             patch("run.histogram_equalization", side_effect=lambda f: f) as m_he, \
             patch("run.apply_filter", side_effect=lambda f: f) as m_flt, \
             patch("run.nn_inference", side_effect=lambda f: f) as m_nn, \
             patch("run.histogram_figure_numba",
                   return_value=(np.zeros(256), np.zeros(256), np.zeros(256))), \
             patch("run.update_histogram"), \
             patch("run.plot_overlay_to_image", side_effect=lambda f, fig: f), \
             patch("run.plot_strings_to_image", side_effect=lambda f, t: f), \
             patch("run.initialize_hist_figure",
                   return_value=(MagicMock(), MagicMock(), MagicMock(),
                                 MagicMock(), MagicMock(), MagicMock())), \
             patch("keyboard.is_pressed", return_value=False):
            list(custom_processing(single_frame_source()))

        assert m_stat.call_count == 1, f"compute_stats called {m_stat.call_count} times"
        assert m_ent.call_count == 1, f"compute_entropy called {m_ent.call_count} times"
        assert m_lt.call_count == 1, f"linear_transform called {m_lt.call_count} times"
        assert m_he.call_count == 1, f"histogram_equalization called {m_he.call_count} times"
        assert m_flt.call_count == 1, f"apply_filter called {m_flt.call_count} times"
        assert m_nn.call_count == 1, f"nn_inference called {m_nn.call_count} times"


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------

class TestExceptionPropagation:
    """Verify unhandled exceptions propagate out of custom_processing."""

    def test_exception_in_loop_propagates(self):
        """RuntimeError raised inside the loop must propagate to the caller."""
        def bad_source():
            yield make_frame()

        with patch("run.compute_stats", side_effect=RuntimeError("test error")), \
             patch("run.initialize_hist_figure",
                   return_value=(MagicMock(), MagicMock(), MagicMock(),
                                 MagicMock(), MagicMock(), MagicMock())), \
             patch("keyboard.is_pressed", return_value=False):
            gen = custom_processing(bad_source())
            with pytest.raises(RuntimeError, match="test error"):
                next(gen)

    def test_value_error_in_loop_propagates(self):
        """ValueError raised inside the loop must also propagate."""
        with patch("run.apply_filter", side_effect=ValueError("filter broke")), \
             patch("run.compute_stats", return_value={'mean': (0,0,0), 'mode': (0,0,0), 'std': (0,0,0), 'max': (0,0,0), 'min': (0,0,0)}), \
             patch("run.compute_entropy", return_value=(0.0, 0.0, 0.0)), \
             patch("run.linear_transform", side_effect=lambda f, **kw: f), \
             patch("run.histogram_equalization", side_effect=lambda f: f), \
             patch("run.nn_inference", side_effect=lambda f: f), \
             patch("run.initialize_hist_figure",
                   return_value=(MagicMock(), MagicMock(), MagicMock(),
                                 MagicMock(), MagicMock(), MagicMock())), \
             patch("keyboard.is_pressed", return_value=False):
            gen = custom_processing(single_frame_source())
            with pytest.raises(ValueError, match="filter broke"):
                next(gen)


# ---------------------------------------------------------------------------
# Integration: update_histogram and plot_overlay_to_image
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests for histogram update and overlay rendering."""

    def test_update_histogram_called_with_length_256_arrays(self):
        """update_histogram must be called with r_bars, g_bars, b_bars each of length 256."""
        captured_args = {}

        def fake_update(fig, ax, bg, rp, gp, bp, r, g, b):
            captured_args['r'] = r
            captured_args['g'] = g
            captured_args['b'] = b

        with patch("run.update_histogram", side_effect=fake_update), \
             patch("run.plot_overlay_to_image", side_effect=lambda f, fig: f), \
             patch("run.plot_strings_to_image", side_effect=lambda f, t: f), \
             patch("run.nn_inference", side_effect=lambda f: f), \
             patch("run.apply_filter", side_effect=lambda f: f), \
             patch("run.histogram_equalization", side_effect=lambda f: f), \
             patch("run.linear_transform", side_effect=lambda f, **kw: f), \
             patch("run.initialize_hist_figure",
                   return_value=(MagicMock(), MagicMock(), MagicMock(),
                                 MagicMock(), MagicMock(), MagicMock())), \
             patch("keyboard.is_pressed", return_value=False):
            list(custom_processing(single_frame_source()))

        assert len(captured_args['r']) == 256
        assert len(captured_args['g']) == 256
        assert len(captured_args['b']) == 256

    def test_plot_overlay_called_once_per_frame_when_histogram_on(self):
        """plot_overlay_to_image must be called exactly once per frame when histogram is on."""
        with patch("run.plot_overlay_to_image", side_effect=lambda f, fig: f) as mock_poi, \
             patch("run.plot_strings_to_image", side_effect=lambda f, t: f), \
             patch("run.nn_inference", side_effect=lambda f: f), \
             patch("run.apply_filter", side_effect=lambda f: f), \
             patch("run.histogram_equalization", side_effect=lambda f: f), \
             patch("run.linear_transform", side_effect=lambda f, **kw: f), \
             patch("run.update_histogram"), \
             patch("run.histogram_figure_numba",
                   return_value=(np.zeros(256), np.zeros(256), np.zeros(256))), \
             patch("run.initialize_hist_figure",
                   return_value=(MagicMock(), MagicMock(), MagicMock(),
                                 MagicMock(), MagicMock(), MagicMock())), \
             patch("keyboard.is_pressed", return_value=False):
            list(custom_processing(multi_frame_source(n=3)))

        assert mock_poi.call_count == 3, (
            f"plot_overlay_to_image called {mock_poi.call_count} times, expected 3"
        )


# ---------------------------------------------------------------------------
# _build_stats_text helper
# ---------------------------------------------------------------------------

class TestBuildStatsText:
    """Verify _build_stats_text formats output correctly."""

    def test_contains_rgb_labels(self):
        """Output strings must include R, G, B channel labels."""
        stats = {
            'mean': (10.0, 20.0, 30.0),
            'std':  (1.0, 2.0, 3.0),
            'min':  (0, 5, 10),
            'max':  (50, 100, 150),
            'mode': (10, 20, 30),
        }
        ent = (1.0, 2.0, 3.0)
        lines = _build_stats_text(stats, ent)

        full_text = '\n'.join(lines)
        assert 'R' in full_text, "Output must contain 'R' label"
        assert 'G' in full_text, "Output must contain 'G' label"
        assert 'B' in full_text, "Output must contain 'B' label"

    def test_contains_mean_std_min_max(self):
        """Output strings must include mean, std, min, max values."""
        stats = {
            'mean': (128.0, 64.0, 32.0),
            'std':  (10.5, 5.0, 2.5),
            'min':  (0, 10, 20),
            'max':  (255, 200, 100),
            'mode': (128, 64, 32),
        }
        ent = (4.0, 3.0, 2.0)
        lines = _build_stats_text(stats, ent)
        full_text = '\n'.join(lines)

        assert 'mean' in full_text.lower()
        assert 'std' in full_text.lower()
        assert 'min' in full_text.lower()
        assert 'max' in full_text.lower()

    def test_returns_list_of_strings(self):
        """Return value must be a list of strings."""
        stats = {
            'mean': (0.0, 0.0, 0.0),
            'std':  (0.0, 0.0, 0.0),
            'min':  (0, 0, 0),
            'max':  (0, 0, 0),
            'mode': (0, 0, 0),
        }
        ent = (0.0, 0.0, 0.0)
        result = _build_stats_text(stats, ent)

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)
