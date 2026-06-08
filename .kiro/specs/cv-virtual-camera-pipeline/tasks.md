# Tasks: cv-virtual-camera-pipeline

## Task List

- [x] 1. Implement `compute_stats` in `basics.py`
  - [x] 1.1 Compute per-channel mean, mode, standard deviation, maximum, and minimum from `frame[:, :, ch]` for ch in {0,1,2}
  - [x] 1.2 Return results as a dict with keys `'mean'`, `'mode'`, `'std'`, `'max'`, `'min'` each mapping to a 3-tuple (R, G, B)
  - [x] 1.3 Write property-based tests in `tests/test_stats.py`:
    - Property 1 test: for any random uint8 frame, mean/mode/max/min in [0,255] and std in [0,127.5]
    - Property 2 test: for any constant-fill frame, std == 0.0 for all channels
  - [x] 1.4 Write example-based unit tests: known constant frames → verify exact values for all five metrics

- [x] 2. Implement `compute_entropy` in `basics.py`
  - [x] 2.1 Compute normalised intensity histogram per channel and apply Shannon entropy formula `H = -Σ p·log2(p)` for non-zero probabilities
  - [x] 2.2 Return `(entropy_r, entropy_g, entropy_b)` as floats in [0.0, 8.0]
  - [x] 2.3 Write property-based tests in `tests/test_entropy.py`:
    - Property 3 test: for any random uint8 frame, all returned entropies are floats in [0.0, 8.0] and two calls return identical results
  - [x] 2.4 Write example-based unit tests: single-intensity channel → 0.0; perfectly uniform channel → 8.0

- [x] 3. Implement `histogram_figure_numba` in `basics.py`
  - [x] 3.1 Fill in the `@njit`-decorated function body using plain loops (no OpenCV, no Python objects) to count pixel intensities per channel into three length-256 arrays `r_bars`, `g_bars`, `b_bars`
  - [x] 3.2 Return `(r_bars, g_bars, b_bars)` as Numba-compatible 1-D integer arrays
  - [x] 3.3 Write property-based tests in `tests/test_histogram.py`:
    - Property 4 test: for any random uint8 frame, all three returned arrays have length 256 and all elements are non-negative
  - [x] 3.4 Write example-based unit tests: frame with all Red pixels == 128 → only `r_bars[128]` is non-zero; verify `@njit` decorator present

- [x] 4. Implement `linear_transform` in `basics.py`
  - [x] 4.1 Apply `clip(alpha * channel + beta, 0, 255)` independently to each channel using NumPy vectorised operations
  - [x] 4.2 Return a `uint8` array of the same shape as the input frame
  - [x] 4.3 Write property-based tests in `tests/test_linear.py`:
    - Property 5 test: for any random uint8 frame, `linear_transform(frame, 1.0, 0.0)` equals the input frame
    - Property 6 test: for any random (frame, alpha, beta), output has same shape, dtype `uint8`, and all values in [0, 255]
  - [x] 4.4 Write example-based unit tests: channel independence (only R channel non-zero → G and B unchanged after transform)

- [x] 5. Implement `histogram_equalization` in `basics.py`
  - [x] 5.1 Apply the standard CDF-based LUT equalization independently to each channel (R, G, B treated separately)
  - [x] 5.2 Return a `uint8` array of the same shape as the input frame
  - [x] 5.3 Write property-based tests in `tests/test_equalization.py`:
    - Property 7 test: for any random uint8 frame, applying equalization twice gives identical results on the second call as on the first
    - Property 8 test: for any random uint8 frame, output has same shape, dtype `uint8`, and all values in [0, 255]
  - [x] 5.4 Write example-based unit tests: verify equalization matches `cv2.equalizeHist` applied per channel

- [x] 6. Implement `apply_filter` in `basics.py`
  - [x] 6.1 Choose one filter from: edge detection, blur, sharpen, Sobel, or Gabor; implement it using OpenCV or NumPy
  - [x] 6.2 Clamp or normalise the result to [0, 255] and cast to `uint8` before returning
  - [x] 6.3 Return a frame of the same spatial dimensions (height × width) as the input
  - [x] 6.4 Write property-based tests in `tests/test_filter.py`:
    - Property 9 test: for any random uint8 frame, output has identical height and width, dtype `uint8`, and all values in [0, 255]
  - [x] 6.5 Write smoke test: call `apply_filter` with a valid frame; verify it runs without exception

- [x] 7. Implement `nn_inference` in `basics.py`
  - [x] 7.1 Select a pre-trained model (face detection, object keypoints, or segmentation) available from OpenCV DNN, MediaPipe, or a bundled weights file
  - [x] 7.2 Load the model at module level (or on first call) and raise `RuntimeError` with a descriptive message if the model file is missing or fails to load
  - [x] 7.3 Run inference on each frame; annotate or modify the frame as appropriate for the chosen task
  - [x] 7.4 If no detection/region is found, return the original frame unmodified
  - [x] 7.5 Ensure the returned frame is `uint8`, same spatial dimensions as input, with all values in [0, 255]
  - [x] 7.6 Write property-based tests in `tests/test_nn.py`:
    - Property 10 test: for any valid uint8 frame, output has same height/width, dtype `uint8`, and all values in [0, 255]
  - [x] 7.7 Write example-based unit tests: model path missing → `RuntimeError` raised; solid-color frame → output equals input

- [x] 8. Wire all processing functions into `custom_processing` in `run.py`
  - [x] 8.1 Import all functions from `basics.py` at the top of `run.py`
  - [x] 8.2 Inside the frame loop, call functions in this order: `compute_stats` → `compute_entropy` → `linear_transform` → `histogram_equalization` → `apply_filter` → `nn_inference` → `histogram_figure_numba` → `update_histogram` → `plot_overlay_to_image` → `_build_stats_text` / `plot_strings_to_image`
  - [x] 8.3 Implement `_build_stats_text` helper that formats the stats dict and entropy tuple into a list of strings with R/G/B labels and mean, std, min, max values
  - [x] 8.4 Guard `plot_strings_to_image` call: skip if `frame.shape[1] < 400` or `frame.shape[0] < 70`
  - [x] 8.5 Implement "h" key toggle for histogram overlay with debounce counter (reset to 10 on press, decrement each frame, only toggle when counter == 0)
  - [x] 8.6 Ensure unhandled exceptions propagate naturally (no bare `except` clauses that swallow them)
  - [x] 8.7 Write integration / pipeline tests in `tests/test_pipeline.py`:
    - Verify `custom_processing` is a generator function that yields one frame per input
    - Verify output frame has shape `(720, 1280, 3)`, dtype `uint8`, values in [0, 255]
    - Verify `plot_strings_to_image` is NOT called when frame width < 400 (mock-based)
    - Verify "h" key toggles histogram with debounce (mock `keyboard.is_pressed`)
    - Verify each processor function is called exactly once per frame in defined order (mock-based)
    - Verify exceptions raised inside the loop propagate out of `custom_processing`
    - Integration test: `update_histogram` called with arrays of length 256 each frame cycle
    - Integration test: `plot_overlay_to_image` called once per frame cycle when histogram is enabled
