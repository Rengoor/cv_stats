# Requirements Document

## Introduction

This feature implements a real-time video processing pipeline for a university Computer Vision project (THI, SS2025). The pipeline captures frames from a physical webcam via OpenCV, applies a set of image processing operations (statistical analysis, transformations, filters, histogram equalization, and neural-network-based inference), and streams the processed frames to a virtual camera device via pyvirtualcam so the result is available to applications such as Zoom or Discord.

All image processing logic lives in `basics.py`. The entry point `run.py` wires the pipeline together through the `custom_processing` generator function. The already-implemented modules `capturing.py` and `overlays.py` must not be modified.

---

## Glossary

- **Pipeline**: The end-to-end data flow from webcam capture through image processing to virtual camera output.
- **Frame**: A single RGB image (NumPy `uint8` array of shape `(H, W, 3)`) produced by the webcam capture loop.
- **Channel**: One of the three colour planes of a Frame — Red (index 0), Green (index 1), or Blue (index 2).
- **Processor**: The collection of functions in `basics.py` that transform or analyse a Frame.
- **Histogram_Function**: The Numba-JIT-compiled function `histogram_figure_numba` in `basics.py`.
- **Neural_Network**: A pre-trained deep-learning model used for the special task (object/face detection, region replacement, or segmentation).
- **VirtualCamera**: The class in `capturing.py` responsible for webcam capture and virtual camera output.
- **Overlay_Module**: The module `overlays.py` providing histogram figure helpers and text rendering.
- **Display_Text**: A list of strings rendered onto the Frame by `plot_strings_to_image`.
- **Mode**: The pixel intensity value (0–255) that appears most frequently in a Channel.
- **Linear_Transformation**: A per-pixel affine operation `output = α · input + β` applied independently to each Channel.
- **Entropy**: The Shannon entropy of the pixel intensity distribution of a Channel.
- **Histogram_Equalization**: The operation that redistributes pixel intensities so that the cumulative histogram of a Channel is approximately uniform.

---

## Requirements

### Requirement 1: Statistical Metrics per Channel

**User Story:** As a student, I want per-channel statistical metrics computed for every frame, so that I can verify the numerical properties of the image and display them as overlays.

#### Acceptance Criteria

1. THE Processor SHALL compute the mean pixel intensity for each Channel of a Frame and return three scalar values (one per Channel).
2. THE Processor SHALL compute the Mode of pixel intensities for each Channel of a Frame and return three scalar values (one per Channel).
3. THE Processor SHALL compute the standard deviation of pixel intensities for each Channel of a Frame and return three scalar values (one per Channel).
4. THE Processor SHALL compute the maximum pixel intensity for each Channel of a Frame and return three scalar values (one per Channel).
5. THE Processor SHALL compute the minimum pixel intensity for each Channel of a Frame and return three scalar values (one per Channel).
6. WHEN a statistical metric function is called with a valid Frame, THE Processor SHALL return values in the range [0, 255] for mean, Mode, maximum, and minimum metrics.
7. WHEN a statistical metric function is called with a valid Frame, THE Processor SHALL return a standard deviation value in the range [0, 127.5].
8. WHEN a Frame is a uniform solid colour, THE Processor SHALL return a standard deviation of 0.0 for each Channel.

---

### Requirement 2: Entropy per Channel

**User Story:** As a student, I want the Shannon entropy computed per channel for every frame, so that I can measure and display the information content of each colour plane.

#### Acceptance Criteria

1. THE Processor SHALL compute the Shannon entropy of pixel intensity distribution for each Channel of a Frame and return three scalar float values (one per Channel).
2. WHEN a Channel contains pixels of a single intensity value, THE Processor SHALL return an entropy of 0.0 for that Channel.
3. WHEN a Channel contains a uniform distribution across all 256 intensity values, THE Processor SHALL return an entropy value of 8.0 bits for that Channel.
4. WHEN entropy is computed twice on the same Frame, THE Processor SHALL return identical values on both calls (deterministic output).

---

### Requirement 3: Histogram Computation

**User Story:** As a student, I want per-channel pixel intensity histograms computed via a JIT-compiled function, so that the histogram overlay runs at real-time frame rates without dropping performance.

#### Acceptance Criteria

1. THE Histogram_Function SHALL accept a single Frame as input and return three arrays — `r_bars`, `g_bars`, `b_bars` — each of length 256.
2. WHEN called with a valid Frame, THE Histogram_Function SHALL return non-negative values in each element of `r_bars`, `g_bars`, and `b_bars`.
3. THE Histogram_Function SHALL be decorated with `@njit` from the numba library to enable JIT compilation.
4. WHEN called with a Frame where all pixels in the Red Channel have intensity 128, THE Histogram_Function SHALL return an `r_bars` array where only index 128 is non-zero.
5. WHEN the Histogram_Function is integrated in the `custom_processing` generator, THE Pipeline SHALL pass `r_bars`, `g_bars`, `b_bars` to `update_histogram` from the Overlay_Module each frame cycle.

---

### Requirement 4: Histogram Overlay Display

**User Story:** As a student, I want a live three-channel histogram rendered onto each output frame, so that I can visually inspect the pixel distribution while the virtual camera is running.

#### Acceptance Criteria

1. WHEN a Frame is processed, THE Pipeline SHALL render the histogram of all three Channels as a single line plot onto the Frame using the Overlay_Module's `plot_overlay_to_image` function.
2. THE Pipeline SHALL render the Red Channel histogram with a red line, the Green Channel histogram with a green line, and the Blue Channel histogram with a blue line on the same axes.
3. THE Pipeline SHALL initialise the histogram matplotlib figure once before the frame loop begins and reuse it for every subsequent frame.

---

### Requirement 5: Linear Transformation

**User Story:** As a student, I want a per-channel linear (affine) transformation applied to frames, so that I can demonstrate brightness and contrast adjustment as a basic image processing operation.

#### Acceptance Criteria

1. THE Processor SHALL implement a linear transformation function that accepts a Frame and per-channel parameters `α` (gain) and `β` (bias), and returns a transformed Frame of the same shape and dtype.
2. WHEN the linear transformation is applied with `α = 1.0` and `β = 0.0`, THE Processor SHALL return a Frame whose pixel values are identical to the input Frame (identity transform).
3. WHEN the linear transformation is applied, THE Processor SHALL clamp all output pixel values to the range [0, 255] before returning the Frame.
4. THE Processor SHALL apply the linear transformation independently to each Channel without mixing Channel data.

---

### Requirement 6: Histogram Equalization

**User Story:** As a student, I want per-channel histogram equalization applied to frames, so that I can demonstrate contrast enhancement as a required image processing operation.

#### Acceptance Criteria

1. THE Processor SHALL implement a histogram equalization function that accepts a Frame and returns an equalized Frame of the same shape and dtype.
2. THE Processor SHALL apply histogram equalization independently to each Channel of the Frame.
3. WHEN histogram equalization is applied to a Frame, THE Processor SHALL return a Frame whose pixel values remain in the range [0, 255].
4. WHEN histogram equalization is applied twice to the same Frame, THE Processor SHALL return a Frame on the second call whose pixel values are identical to the result of the first call (idempotent after first application).

---

### Requirement 7: Image Filter

**User Story:** As a student, I want at least one non-trivial image filter applied to frames, so that I can demonstrate spatial frequency processing as a required image processing operation.

#### Acceptance Criteria

1. THE Processor SHALL implement at least one image filter from the following set: edge detection, blur, sharpen, Sobel, or Gabor.
2. WHEN a filter function is called with a valid Frame, THE Processor SHALL return a Frame of the same spatial dimensions (height × width) as the input.
3. WHEN a filter function is called with a valid Frame, THE Processor SHALL return a Frame whose pixel values are in the range [0, 255] and whose dtype is `uint8`.
4. IF a filter produces a result outside the [0, 255] range, THEN THE Processor SHALL clamp or normalise the result to [0, 255] before returning the Frame.

---

### Requirement 8: Neural Network Special Task

**User Story:** As a student, I want a neural-network-based processing step integrated into the pipeline, so that I fulfill the special task requirement of the assignment using deep learning inference.

#### Acceptance Criteria

1. THE Processor SHALL implement a neural network inference function that processes a Frame using a pre-trained Neural_Network model loaded from disk or a library.
2. THE Neural_Network SHALL perform one of the following tasks on each Frame: face/object keypoint detection, replacement of a detected region with an alternate image, or semantic image segmentation.
3. WHEN the Neural_Network inference function is called with a valid Frame, THE Processor SHALL return a Frame of the same spatial dimensions as the input Frame.
4. WHEN the Neural_Network model file is not found or fails to load, THE Processor SHALL raise a descriptive `RuntimeError` identifying the missing resource.
5. WHEN no detection or segmentation region is found in a Frame, THE Processor SHALL return the original Frame unmodified.
6. WHEN the Neural_Network inference function is called with a valid Frame, THE Processor SHALL return a Frame whose pixel values are in the range [0, 255] and whose dtype is `uint8`.

---

### Requirement 9: Text Overlay of Statistics

**User Story:** As a student, I want computed statistics displayed as text on each output frame, so that I can observe numerical values alongside the visual output in real time.

#### Acceptance Criteria

1. WHEN a Frame is processed, THE Pipeline SHALL pass a list of formatted strings containing at least mean, standard deviation, minimum, and maximum values for each Channel to `plot_strings_to_image` from the Overlay_Module.
2. WHEN the statistics text list is rendered, THE Pipeline SHALL include Channel labels (e.g., "R:", "G:", "B:") to identify which values belong to which Channel.
3. WHEN the image dimensions are smaller than the minimum required by `plot_strings_to_image` (width < 400 px or height < 70 px), THE Pipeline SHALL skip text rendering for that frame and continue processing.

---

### Requirement 10: Pipeline Integration and Frame Loop

**User Story:** As a student, I want the processing pipeline wired correctly in `run.py`, so that every frame passes through all processing steps and is delivered to the virtual camera in the correct format.

#### Acceptance Criteria

1. THE Pipeline SHALL implement `custom_processing` as a Python generator function that accepts an image source generator and yields a processed Frame for each input Frame.
2. WHEN the `custom_processing` generator yields a Frame, THE Frame SHALL be a NumPy array of shape `(720, 1280, 3)` with dtype `uint8` and values in the range [0, 255].
3. WHEN the user presses the "q" key, THE VirtualCamera SHALL stop capturing frames and the pipeline SHALL terminate cleanly without raising an unhandled exception.
4. WHEN the user presses a configurable hotkey (e.g., "h"), THE Pipeline SHALL toggle the histogram overlay on or off for subsequent frames.
5. THE Pipeline SHALL call each Processor function once per Frame in a defined, deterministic order.
6. WHEN an unhandled exception occurs inside `custom_processing`, THE Pipeline SHALL allow the exception to propagate so the runtime can report it to the user.
