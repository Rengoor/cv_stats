# cv_stats — Complete Project Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Dependencies](#3-dependencies)
4. [Module Reference](#4-module-reference)
   - [basics.py](#41-basicspy)
   - [capturing.py](#42-capturingpy)
   - [overlays.py](#43-overlayspy)
   - [emotion_detection.py](#44-emotion_detectionpy)
   - [run.py](#45-runpy)
5. [Test Suite](#5-test-suite)
   - [test_stats.py](#51-test_statspy)
   - [test_entropy.py](#52-test_entropypy)
   - [test_histogram.py](#53-test_histogrampy)
   - [test_linear.py](#54-test_linearpy)
   - [test_equalization.py](#55-test_equalizationpy)
   - [test_filter.py](#56-test_filterpy)
   - [test_nn.py](#57-test_nnpy)
   - [test_pipeline.py](#58-test_pipelinepy)
6. [Hotkey Reference](#6-hotkey-reference)
7. [Pipeline Execution Flow](#7-pipeline-execution-flow)
8. [Running the Application](#8-running-the-application)
9. [Running the Tests](#9-running-the-tests)

---

## 1. Project Overview

`cv_stats` is a real-time computer vision pipeline that reads frames from a webcam, applies a chain of image processing operations, and outputs the result to a virtual camera device (or a local preview window when no virtual camera is available).

Key capabilities:

- **Per-frame image statistics** — mean, mode, standard deviation, min, max, and Shannon entropy computed per RGB channel.
- **Histogram visualisation** — a live RGB histogram overlaid in the top-left corner.
- **Linear (affine) contrast enhancement** — configurable gain and bias applied per channel.
- **Histogram equalisation** — CDF-based equalisation per channel, reusing pre-computed histogram bins for speed.
- **Gaussian blur filter** — 21×21 kernel smoothing applied pre- and post-equalisation.
- **Background replacement** — MediaPipe Selfie Segmentation separates the person from the background and replaces the background with a strongly blurred version.
- **Emotion detection with mood UI** — the FER library detects facial emotion, chooses a procedural gradient background that matches the detected mood, and overlays an info panel and optional smiley graphic.
- **Virtual camera output** — the processed frame is sent to a `pyvirtualcam` virtual camera so it appears as a camera source in other applications (e.g. video-conferencing software).
- **Interactive hotkeys** — four features can be toggled at runtime without restarting the application.

---

## 2. Architecture

```
run.py  (entry point + main processing loop)
  │
  ├── capturing.py      VirtualCamera class — screen/webcam capture + virtual cam output
  ├── basics.py         Core image-processing primitives (histogram, stats, transforms)
  ├── overlays.py       Matplotlib histogram figure + text/image overlay utilities
  └── emotion_detection.py   EmotionDetector + mood background + emotion UI overlay
```

`run.py` is the orchestrator. It wires the generator pipeline together:

```
webcam frames  →  custom_processing()  →  virtual camera / preview window
```

`custom_processing` is itself a Python generator — it consumes frames one at a time from the upstream source generator and yields one processed frame per input frame.

---

## 3. Dependencies

All pinned versions are listed in `requirements.txt`.

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.23.5 | Array maths throughout |
| opencv-python | 4.7.0.72 | Gaussian blur, equalisation, text rendering |
| pyvirtualcam | 0.10.2 | Write frames to a virtual camera device |
| pillow | 9.4.0 | Screen capture via `ImageGrab` |
| matplotlib | 3.7.1 | Live histogram figure |
| numba | 0.56.4 | JIT-compiled histogram loop |
| moviepy | 1.0.3 | (available, not used in the active pipeline) |
| keyboard | 0.13.5 | Global hotkey detection |

Optional (not in requirements.txt, must be installed separately):

| Package | Purpose |
|---------|---------|
| mediapipe | Selfie segmentation (background replacement) and emotion path segmentation |
| fer | Facial emotion recognition |

If either optional package is unavailable the corresponding feature is silently disabled and the pipeline continues without it.

---

## 4. Module Reference

### 4.1 `basics.py`

Core image-processing library. All functions operate on NumPy `uint8` arrays with shape `(H, W, 3)` and channel order R, G, B unless otherwise noted.

---

#### `histogram_figure_numba(np_img) → (r_bars, g_bars, b_bars)`

Computes a per-channel pixel intensity histogram using a Numba JIT-compiled loop.

- **Decorator**: `@njit` — compiled to native machine code on first call.
- **Input**: `np_img` — `uint8` array `(H, W, 3)`.
- **Returns**: three `int64` arrays of length 256, one per channel (R, G, B), where `bars[i]` is the count of pixels with intensity value `i`.
- **Performance note**: Uses explicit Python-style loops instead of NumPy vectorisation because `@njit` eliminates their overhead while keeping the code compatible with Numba's supported NumPy subset.

---

#### `compute_stats_and_entropy_from_hist(r_bars, g_bars, b_bars, total_pixels) → (stats, entropies)`

Computes five statistical metrics and Shannon entropy for each channel from pre-computed histogram arrays.

- **Inputs**:
  - `r_bars`, `g_bars`, `b_bars` — histogram arrays from `histogram_figure_numba`.
  - `total_pixels` — integer, `H * W`, used as the normalisation denominator.
- **Returns**:
  - `stats` — `dict` with keys `'mean'`, `'mode'`, `'std'`, `'max'`, `'min'`, each a 3-tuple `(R, G, B)`.
  - `entropies` — 3-tuple `(R_entropy, G_entropy, B_entropy)` in bits (base-2 logarithm), range `[0.0, 8.0]`.

Metric definitions:

| Metric | Formula |
|--------|---------|
| mean | `Σ(i * count[i]) / total_pixels` |
| mode | `argmax(bars)` — intensity with the highest count |
| std | `sqrt(Σ(count[i] * (i - mean)²) / total_pixels)` |
| min | smallest intensity `i` where `count[i] > 0` |
| max | largest intensity `i` where `count[i] > 0` |
| entropy | `-Σ(p * log₂(p))` over non-zero probabilities `p = count[i] / total_pixels` |

---

#### `linear_transform(frame, alpha, beta) → np.ndarray`

Applies a per-channel affine transformation: `output = clip(alpha * input + beta, 0, 255)`.

- **Inputs**:
  - `frame` — `uint8` `(H, W, 3)`.
  - `alpha` — `float`, multiplicative gain.
  - `beta` — `float`, additive bias.
- **Returns**: `uint8` array, same shape as input. Computation uses `float64` intermediate to avoid overflow before clamping.
- No channel mixing occurs; each channel is transformed independently.

---

#### `apply_filter(frame) → np.ndarray`

Applies a 21×21 Gaussian blur (sigma=1.0) to the frame using `cv2.GaussianBlur`.

- **Input**: `uint8` `(H, W, 3)`.
- **Returns**: `uint8`, same spatial dimensions. Output is clamped to `[0, 255]`.

---

#### `segment_background(frame, blur_ksize=55) → np.ndarray`

Replaces the frame background with a blurred copy using MediaPipe Selfie Segmentation.

- **Input**: `uint8` RGB `(H, W, 3)`.
- **Parameter**: `blur_ksize` — kernel size for background blur (must be odd; auto-incremented if even).
- **Returns**: `uint8` `(H, W, 3)`. If MediaPipe is not installed, returns the original frame unchanged.

Pipeline steps:
1. Run `SelfieSegmentation` → soft probability mask `(H, W)` float32.
2. Threshold at 0.5 → binary person/background mask.
3. Create a strongly blurred copy of the frame.
4. Composite: person pixels from original, background pixels from blurred copy.

The model (`model_selection=0`, general/fast model) is loaded once at module import time into module-level variable `_selfie_seg`. The flag `_SEGMENTATION_AVAILABLE` is `False` if the import fails.

---

#### `histogram_equalization(frame, r_bars, g_bars, b_bars) → np.ndarray`

Applies CDF-based histogram equalisation per channel, reusing pre-computed histogram bins.

- **Inputs**: `uint8` `(H, W, 3)` frame plus the three histogram arrays from `histogram_figure_numba`.
- **Returns**: `uint8` `(H, W, 3)`.

Algorithm per channel:
1. Compute the cumulative distribution function (CDF) of the histogram.
2. Build a 256-entry look-up table (LUT): `LUT[i] = round((CDF[i] - CDF_min) / (total_pixels - CDF_min) * 255)`.
3. Apply the LUT via NumPy fancy indexing.

Edge case: if all pixels share the same intensity (`CDF_min == total_pixels`), the LUT is all zeros.

This function is **idempotent**: applying it twice to the same frame yields the same result as applying it once.

---

### 4.2 `capturing.py`

Provides the `VirtualCamera` class, which wraps screen capture, webcam capture, and virtual camera output.

---

#### `class VirtualCamera(fps, width, height)`

| Attribute | Type | Description |
|-----------|------|-------------|
| `fps` | int | Target frames per second |
| `width` | int | Frame width in pixels |
| `height` | int | Frame height in pixels |

---

#### `VirtualCamera.capture_screen(plt_inside=False, alt_width=0, alt_height=0)`

Generator that yields frames captured from the primary monitor using `PIL.ImageGrab.grab`.

- `plt_inside=True` — renders each frame in a Matplotlib window (very slow, for debugging only).
- `alt_width` / `alt_height` — override the default width/height for the capture bounding box.
- Yields `uint8` RGB arrays.

---

#### `VirtualCamera.capture_cv_video(camera_id, bgr_to_rgb=False)`

Generator that yields frames from an OpenCV-compatible camera.

- Opens `cv2.VideoCapture(camera_id)` and configures it with the instance's `width`, `height`, `fps`, and MJPEG codec.
- Prints actual camera properties (may differ from requested values) to stdout.
- `bgr_to_rgb=True` — reverses channel order from OpenCV's default BGR to RGB.
- Stops when the `q` key is pressed; releases the capture device before returning.
- Raises `RuntimeError` if the device cannot be opened or a frame cannot be read.

---

#### `VirtualCamera.virtual_cam_interaction(img_generator, print_fps=True)`

Reads frames from `img_generator` and sends them to a `pyvirtualcam.Camera`.

- Prints "Quit camera stream with 'q'" before entering the loop.
- Calls `cam.sleep_until_next_frame()` after each `cam.send(img)` to maintain the target FPS.

---

### 4.3 `overlays.py`

Matplotlib-based overlay utilities for the histogram figure and text rendering.

---

#### `initialize_hist_figure() → (fig, ax, background, r_plot, g_plot, b_plot)`

Creates and returns a Matplotlib figure pre-configured for efficient animated updates.

- Y-axis fixed at `[0, 3]` — histogram data should be normalised to this range before plotting.
- X-axis covers `[-0.5, 255.5]` (all 256 intensity values).
- Uses `animated=True` on the three line artists to support blitting.
- Returns the figure, axes, a cached background snapshot, and the three line plot objects.

---

#### `update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars)`

Updates the histogram figure in-place using blitting for performance.

Steps: restore the cached background → set new Y data on each line → redraw each artist → blit the axes bounding box.

---

#### `plot_overlay_to_image(np_img, plt_figure) → np.ndarray`

Composites a Matplotlib figure onto a NumPy image array.

- Reads the figure's RGBA pixel buffer and converts to RGB.
- Only non-white pixels are copied onto `np_img`, creating a transparency effect.
- Modifies `np_img` in-place and also returns it.

---

#### `plot_strings_to_image(np_img, list_of_string, text_color=(255,0,0), right_space=400, top_space=50) → np.ndarray`

Renders a list of strings onto the image, stacked vertically starting from the top-right area.

- Strings are drawn at `x = width - right_space`, starting at `y = top_space`, with `line_height = 20` pixels between lines.
- Uses `cv2.UMat` for GPU-accelerated text rendering where available.
- Raises `Exception` if the frame is too narrow (`w < right_space`) or too short.

---

### 4.4 `emotion_detection.py`

Facial emotion recognition, mood-driven background generation, and the emotion UI overlay.

---

#### Module-level constants

**`EMOTION_INFO`** — maps the seven FER emotion labels to an ASCII symbol and a mood category:

| Emotion | Symbol | Mood |
|---------|--------|------|
| happy | `:)` | Positive |
| surprise | `:O` | Positive |
| neutral | `:\|` | Normal |
| sad | `:(` | Stressed |
| angry | `>:(` | Stressed |
| fear | `D:` | Stressed |
| disgust | `:S` | Stressed |

**`MOOD_BG_RECIPE`** — maps mood categories to background style names:

| Mood | Style | Colours |
|------|-------|---------|
| Positive | cheerful | Deep orange top → bright yellow bottom |
| Normal | clean | Pale blue top → light grey bottom |
| Stressed | calm | Deep teal top → mint green bottom |

---

#### `build_mood_bg(height, width, style) → np.ndarray`

Generates a procedural vertical gradient background image.

- **Returns**: `uint8` RGB `(height, width, 3)`.
- Interpolates one row of colour per pixel vertically from `top` colour to `bottom` colour, then tiles across the full width.
- `style` must be `"cheerful"`, `"clean"`, or `"calm"`. Any other value falls back to the "calm" palette.

---

#### `class EmotionDetector(run_every_n_frames=5)`

Wraps the FER (Facial Emotion Recognition) library with lazy loading and frame-rate throttling.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `run_every_n_frames` | 5 | How often the neural net actually runs. Results are cached between runs. At 30 fps with the default, the net runs ~6 times per second. |

**`EmotionDetector.analyse(rgb_frame) → dict | None`**

The public API method.

- On the first call, imports and initialises the FER model (lazy loading). Prints a success or failure message to stdout.
- Returns `None` if FER is not installed or if no face is detected.
- Returns a `dict` with:

| Key | Type | Description |
|-----|------|-------------|
| `emotion` | str | Dominant emotion label (e.g. `"happy"`) |
| `confidence` | float | Confidence as a percentage (0–100) |
| `symbol` | str | ASCII face symbol |
| `mood` | str | Mood category (`"Positive"`, `"Normal"`, or `"Stressed"`) |

When multiple faces are detected FER picks the one with the largest bounding-box area (most prominent face). Internally, FER receives a BGR image converted from the RGB input.

---

#### `apply_emotion_overlay(frame, seg_mask, result, mood_bg) → np.ndarray`

Composites the emotion UI onto a frame.

- **Inputs**:
  - `frame` — `uint8` RGB `(H, W, 3)`.
  - `seg_mask` — `float32` `(H, W)` probability mask from MediaPipe, or `None`.
  - `result` — emotion dict from `EmotionDetector.analyse`, or `None`.
  - `mood_bg` — gradient background from `build_mood_bg`, or `None`.
- **Returns**: composited `uint8` RGB frame.

Rendering steps:
1. **Background replacement** (if `seg_mask` and `mood_bg` are both provided): replaces background pixels (mask < 0.5) with the mood gradient.
2. **Smiley face** (Positive mood only): drawn at the top-right corner of a blank canvas.
3. **Info panel**: dark semi-transparent rectangle containing emotion label, confidence %, mood label (colour-coded), and background style name.
4. The canvas (smiley + panel) is **horizontally flipped** before compositing onto the frame. This mirrors only the overlay, not the video image.
5. Non-black canvas pixels are alpha-blended at 85% opacity onto the frame.

---

### 4.5 `run.py`

Entry point and main processing loop.

---

#### `_init_hist_figure(fig_w_px=300, fig_h_px=180, dpi=80) → (fig, ax, r_plot, g_plot, b_plot)`

Creates a dark-themed Matplotlib histogram figure sized to `fig_w_px × fig_h_px` pixels.

- Background colour `#111111`, tick colour white, spine colour `#444444`.
- Returns the figure, axes, and three line artists (R, G, B). Note: unlike `overlays.initialize_hist_figure` this version does **not** return a cached background snapshot because blitting is not used here.

---

#### `_normalise_bars(bars, scale=2.8) → np.ndarray`

Scales histogram bin counts so that the peak bin equals `scale` (fitting the `[0, 3]` Y-axis). Returns a `float64` array. Safe against all-zero input.

---

#### `_overlay_hist_figure(frame, fig) → np.ndarray`

Renders the Matplotlib figure into an RGB array and alpha-blends it into the top-left corner of `frame` at 85% opacity. Automatically clamps to frame dimensions if the figure is larger than the frame.

---

#### `_draw_stats(frame, stats, entropy) → np.ndarray`

Draws the statistics panel at the bottom of `frame`.

- Renders text onto a blank canvas (same size as frame), flips the canvas horizontally, then composites non-black pixels at 80% opacity onto the frame.
- Three rows for R, G, B channels showing mean, mode, std, min, max.
- One row for Shannon entropy per channel in bits.

---

#### `custom_processing(img_source_generator)`

The main processing generator. Consumes frames from `img_source_generator` and yields one processed frame per input.

**Per-frame pipeline** (in order):

| Step | Operation | Condition |
|------|-----------|-----------|
| 1 | `histogram_figure_numba` on raw frame → `r_bars_raw`, `g_bars_raw`, `b_bars_raw` | Always |
| 2 | `compute_stats_and_entropy_from_hist` → `stats`, `ent` | Always |
| 3 | `apply_filter` (blur) on working frame | `apply_blur == True` |
| 4 | `segment_background` | `apply_bg_replacement == True` |
| 5 | `linear_transform(alpha=1.1, beta=5.0)` | Always |
| 6 | Recompute histogram on working frame | `apply_blur == True`; otherwise reuse raw bars |
| 7 | `histogram_equalization` | Always |
| 8 | `apply_filter` (blur again) | `apply_blur == True` |
| 9 | `emotion_detector.analyse` + `apply_emotion_overlay` | `apply_emotion == True` |
| 10 | `_overlay_hist_figure` | `show_histogram == True` |
| 11 | `_draw_stats` | Always |
| 12 | Hotkey polling and debounce | Always |

**Initial toggle states**:

| Toggle | Default |
|--------|---------|
| `show_histogram` | `True` |
| `apply_blur` | `True` |
| `apply_bg_replacement` | `False` |
| `apply_emotion` | `False` |

**Debounce**: all four hotkeys use a 15-frame cooldown counter to prevent a single key press from firing multiple times.

The emotion-path segmentor (`_emo_seg`) is a separate `SelfieSegmentation` instance created once before the frame loop; it is used only when `apply_emotion` is `True`.

Mood backgrounds are cached in `mood_bg_cache` (keyed by style name) so the gradient image is only generated once per style.

---

#### `main()`

Configures a 640×480 30 fps `VirtualCamera` and starts the pipeline.

- Primary path: virtual camera output via `pyvirtualcam`.
- Fallback: if a `RuntimeError` mentioning `"virtual camera"` or `"backend"` is raised, opens a local `cv2.imshow` preview window instead. Press `q` in the window to quit.

---

## 5. Test Suite

All tests live in the `tests/` directory. They use `pytest`, `hypothesis` (property-based testing), and `unittest.mock`.

Each test module adds the project root to `sys.path` so source modules can be imported without installation.

---

### 5.1 `test_stats.py`

Tests `compute_stats` from `basics.py`.

**Property-based tests**:

| Property | Description |
|----------|-------------|
| Property 1 | For any random `uint8` frame, `mean`/`mode`/`max`/`min` ∈ [0, 255] and `std` ∈ [0, 127.5]. |
| Property 2 | For any constant-fill frame (all pixels the same value), `std == 0.0` for all three channels. |

**Example-based tests** (`TestComputeStatsConstantFrames`):

| Test | Frame | Expected |
|------|-------|----------|
| `test_all_zeros_frame` | All 0 | mean=0, mode=0, std=0, max=0, min=0 per channel |
| `test_all_255_frame` | All 255 | mean=255, mode=255, std=0, max=255, min=255 |
| `test_constant_128_frame` | All 128 | mean=128, mode=128, std=0, max=128, min=128 |
| `test_per_channel_constants` | R=10, G=100, B=200 | Each channel's metrics equal its fill value |
| `test_return_dict_keys` | Any | Result keys == `{'mean','mode','std','max','min'}` |
| `test_return_values_are_3_tuples` | Any | Each value is a 3-element sequence |

---

### 5.2 `test_entropy.py`

Tests `compute_entropy` from `basics.py`.

**Property-based tests**:

| Property | Description |
|----------|-------------|
| Property 3 | For any random `uint8` frame, all entropies are `float` in [0.0, 8.0] and the function is deterministic (two calls with the same input return equal results). |

**Example-based tests** (`TestComputeEntropyExamples`):

| Test | Frame | Expected entropy |
|------|-------|-----------------|
| `test_single_intensity_channel_returns_zero_entropy` | All 128 | 0.0 per channel |
| `test_single_intensity_zero_channel_returns_zero_entropy` | All 0 | 0.0 per channel |
| `test_single_intensity_255_channel_returns_zero_entropy` | All 255 | 0.0 per channel |
| `test_uniform_distribution_returns_8_bits_entropy` | Perfectly uniform (each of 256 values appears equally) | 8.0 per channel |
| `test_returns_three_floats` | Any | 3-tuple of `float` |
| `test_per_channel_independence` | R=10, G=100, B=200 constant | 0.0 per channel |

---

### 5.3 `test_histogram.py`

Tests `histogram_figure_numba` from `basics.py`.

**Property-based tests**:

| Property | Description |
|----------|-------------|
| Property 4 | For any random `uint8` frame, all three returned arrays have length 256 and all elements are non-negative integers. |

**Example-based tests** (`TestHistogramFigureNumbaExamples`):

| Test | Description |
|------|-------------|
| `test_all_red_pixels_128_only_r_bars_128_nonzero` | 8×8 frame with R=128, G=0, B=0: `r_bars[128]` is the only non-zero R bin; G and B only have `bars[0]` non-zero. |
| `test_r_bars_128_count_equals_pixel_count` | `r_bars[128]` must equal `H * W` when all R pixels are 128. |
| `test_njit_decorator_present` | Verifies the function is a Numba `CPUDispatcher` (i.e. `@njit` was applied). |
| `test_total_counts_equal_pixel_count` | Sum of each bar array must equal `H * W`. |
| `test_returns_three_length_256_arrays` | Return value is a 3-tuple of length-256 arrays. |

---

### 5.4 `test_linear.py`

Tests `linear_transform` from `basics.py`.

**Property-based tests**:

| Property | Description |
|----------|-------------|
| Property 5 | `linear_transform(frame, 1.0, 0.0)` returns pixel values identical to the input (identity). |
| Property 6 | For any finite `alpha` and `beta`, output has same shape, `uint8` dtype, all values in [0, 255]. |

Alpha strategy: `float` in [-10.0, 10.0]. Beta strategy: `float` in [-255.0, 255.0].

**Example-based tests** (`TestLinearTransformExamples`):

| Test | Description |
|------|-------------|
| `test_channel_independence_g_and_b_unchanged_when_r_only_nonzero` | G and B start at 0; after transform they equal `clip(beta, 0, 255)`, not influenced by R. |
| `test_channel_independence_r_channel_unaffected_by_g_values` | R output is identical regardless of G and B values. |
| `test_clamping_above_255` | `2.0 * 200 + 100 = 500` → clamped to 255. |
| `test_clamping_below_zero` | `1.0 * 50 - 100 = -50` → clamped to 0. |
| `test_returns_uint8_dtype` | Output dtype is always `uint8`. |
| `test_output_shape_matches_input` | Output shape equals input shape. |
| `test_alpha_zero_produces_constant_beta_frame` | `alpha=0` collapses all pixels to `clip(beta, 0, 255)`. |

---

### 5.5 `test_equalization.py`

Tests `histogram_equalization` from `basics.py`.

**Property-based tests**:

| Property | Description |
|----------|-------------|
| Property 7 | `histogram_equalization` is idempotent: `f(f(x)) == f(x)` for any `uint8` frame. |
| Property 8 | Output has same shape, `uint8` dtype, all values in [0, 255]. |

PBT uses 32×32 frames for speed.

**Example-based tests** (`TestHistogramEqualizationExamples`):

| Test | Description |
|------|-------------|
| `test_matches_cv2_equalize_hist_random_frame` | Output must match `cv2.equalizeHist` applied independently per channel. |
| `test_matches_cv2_equalize_hist_gradient_frame` | Same check with a linear gradient input. |
| `test_output_shape_matches_input` | Shape preserved. |
| `test_output_dtype_is_uint8` | dtype always `uint8`. |
| `test_constant_frame_does_not_crash` | Constant-value frame must not raise. |
| `test_all_zeros_frame` | All-zero input matches cv2 reference. |
| `test_all_max_frame` | All-255 input matches cv2 reference. |
| `test_idempotent_on_concrete_frame` | Concrete idempotency check with a seeded random frame. |

---

### 5.6 `test_filter.py`

Tests `apply_filter` from `basics.py`.

**Property-based tests**:

| Property | Description |
|----------|-------------|
| Property 9 | Output has identical height and width, `uint8` dtype, all values in [0, 255]. |

**Example-based tests** (`TestApplyFilterExamples`):

| Test | Description |
|------|-------------|
| `test_smoke_valid_frame_no_exception` | Must not raise on a valid 720×1280 frame. |
| `test_output_shape_matches_input` | Shape preserved. |
| `test_output_dtype_is_uint8` | dtype always `uint8`. |
| `test_output_values_in_range` | All values in [0, 255]. |
| `test_constant_frame_no_exception` | Constant frame must not raise. |
| `test_all_zeros_frame` | Blur of all-zeros is all-zeros. |
| `test_all_max_frame` | Blur of all-255 is all-255. |

---

### 5.7 `test_nn.py`

Tests `nn_inference` from `basics.py`.

**Property-based tests**:

| Property | Description |
|----------|-------------|
| Property 10 | Output has same height and width, `uint8` dtype, all values in [0, 255]. |

PBT uses 64×64 frames with `max_examples=50` and `deadline=None`.

**Example-based tests** (`TestNNInferenceExamples`):

| Test | Description |
|------|-------------|
| `test_model_loads_without_error` | Import + first call must not raise. |
| `test_solid_colour_frame_returns_input_unchanged` | Solid-colour frame → no face detected → returns input unmodified. |
| `test_all_zeros_frame_returns_input` | Same as above for all-zero frame. |
| `test_all_max_frame_returns_input` | Same for all-255 frame. |
| `test_output_shape_matches_input` | Shape preserved. |
| `test_output_dtype_is_uint8` | dtype always `uint8`. |

---

### 5.8 `test_pipeline.py`

Integration and unit tests for `custom_processing` and helpers in `run.py`.

Hardware-dependent modules (`pyvirtualcam`, `keyboard`, `capturing`) are stubbed out with `unittest.mock` before `run` is imported.

**Generator behaviour** (`TestCustomProcessingGenerator`):

| Test | Description |
|------|-------------|
| `test_is_generator_function` | `custom_processing` must be a generator function. |
| `test_yields_one_frame_per_input` | N input frames → N output frames. |
| `test_output_frame_shape` | Output shape is (720, 1280, 3). |
| `test_output_frame_dtype_uint8` | dtype is `uint8`. |
| `test_output_frame_values_in_range` | All values in [0, 255]. |

**Text overlay size guard** (`TestTextOverlayGuard`):

| Test | Description |
|------|-------------|
| `test_plot_strings_not_called_when_width_less_than_400` | `plot_strings_to_image` must NOT be called for a 300-pixel-wide frame. |
| `test_plot_strings_called_when_frame_is_full_size` | Must be called exactly once for a full 1280×720 frame. |

**Histogram toggle debounce** (`TestHistogramToggleDebounce`):

| Test | Description |
|------|-------------|
| `test_histogram_visible_by_default` | Histogram overlay is rendered on all 15 frames when no key is pressed. |
| `test_h_press_disables_histogram` | Pressing `h` on frame 0 disables the histogram; fewer than 15 overlay renders occur. |
| `test_debounce_prevents_double_toggle` | Holding `h` for 5 frames behaves identically to a single press within the cooldown window. |

**Processor call order** (`TestProcessorCallOrder`):

| Test | Description |
|------|-------------|
| `test_each_processor_called_once_per_frame` | `compute_stats`, `compute_entropy`, `linear_transform`, `histogram_equalization`, `apply_filter`, and `nn_inference` are each called exactly once per frame. |

**Exception propagation** (`TestExceptionPropagation`):

| Test | Description |
|------|-------------|
| `test_exception_in_loop_propagates` | `RuntimeError` raised inside the loop propagates to the caller. |
| `test_value_error_in_loop_propagates` | `ValueError` raised inside the loop also propagates. |

**Integration** (`TestIntegration`):

| Test | Description |
|------|-------------|
| `test_update_histogram_called_with_length_256_arrays` | `update_histogram` receives `r_bars`, `g_bars`, `b_bars` each of length 256. |
| `test_plot_overlay_called_once_per_frame_when_histogram_on` | `plot_overlay_to_image` is called exactly once per input frame when histogram is enabled. |

**`_build_stats_text` helper** (`TestBuildStatsText`):

| Test | Description |
|------|-------------|
| `test_contains_rgb_labels` | Output strings contain 'R', 'G', 'B'. |
| `test_contains_mean_std_min_max` | Output contains 'mean', 'std', 'min', 'max'. |
| `test_returns_list_of_strings` | Return type is `list[str]`. |

---

## 6. Hotkey Reference

All hotkeys use a 15-frame debounce to prevent repeated toggles from a single key press.

| Key | Feature | Default State |
|-----|---------|---------------|
| `h` | RGB histogram overlay | On |
| `b` | Gaussian blur (pre- and post-equalisation) | On |
| `s` | MediaPipe background replacement | Off |
| `e` | FER emotion detection + mood UI | Off |
| `q` | Quit camera stream (in `capture_cv_video`) | — |

---

## 7. Pipeline Execution Flow

```
                          ┌─────────────────────────────────────────┐
                          │           custom_processing()            │
                          │                                          │
  Raw frame (RGB)  ──────►│  1. histogram_figure_numba               │
                          │  2. compute_stats_and_entropy_from_hist  │
                          │  3. [b] apply_filter (blur)              │
                          │  4. [s] segment_background               │
                          │  5. linear_transform (α=1.1, β=5.0)     │
                          │  6. histogram_figure_numba (if blur on)  │
                          │  7. histogram_equalization               │
                          │  8. [b] apply_filter (blur again)        │
                          │  9. [e] EmotionDetector.analyse          │
                          │         + apply_emotion_overlay          │
                          │  10.[h] _overlay_hist_figure             │
                          │  11.    _draw_stats                      │
                          │  12.    hotkey polling                   │
                          │                                          │
  Processed frame  ◄──────│  yield working_frame                     │
                          └─────────────────────────────────────────┘
                                          │
                          ┌──────────────▼──────────────┐
                          │   pyvirtualcam.Camera.send   │
                          │   (or cv2.imshow fallback)   │
                          └─────────────────────────────┘
```

---

## 8. Running the Application

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. (Optional) Install MediaPipe and FER for background replacement and emotion detection:
   ```
   pip install mediapipe fer
   ```

3. Run:
   ```
   python run.py
   ```

The application will attempt to use a virtual camera. If no virtual camera backend is found it falls back to a local `cv2.imshow` preview window. Press `q` to quit.

---

## 9. Running the Tests

```
pytest tests/
```

Property-based tests use Hypothesis with `max_examples=100` by default. To run faster during development:

```
pytest tests/ --hypothesis-seed=0 -x
```

To run a specific test module:

```
pytest tests/test_stats.py -v
```
