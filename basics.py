# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:18:29 2021

@author: droes
"""
import numpy as np
from numba import njit # conda install numba

@njit
def histogram_figure_numba(np_img):
    '''
    Jit compiled function to increase performance.
    Use some loops insteads of purely numpy functions.
    If you face some compile errors using @njit, see: https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html
    In case you dont need performance boosts, remove the njit flag above the function
    Do not use cv2 functions together with @njit
    '''
    r_bars = np.zeros(256, dtype=np.int64)
    g_bars = np.zeros(256, dtype=np.int64)
    b_bars = np.zeros(256, dtype=np.int64)

    for h in range(np_img.shape[0]):
        for w in range(np_img.shape[1]):
            r_bars[np_img[h, w, 0]] += 1
            g_bars[np_img[h, w, 1]] += 1
            b_bars[np_img[h, w, 2]] += 1

    return r_bars, g_bars, b_bars



####

### All other basic functions

####

from scipy.stats import mode as scipy_mode


def compute_stats(frame: np.ndarray) -> dict:
    """
    Compute per-channel statistical metrics for an RGB frame.

    Args:
        frame: NumPy uint8 array of shape (H, W, 3), channels ordered R, G, B.

    Returns:
        dict with keys 'mean', 'mode', 'std', 'max', 'min', each mapping to a
        3-tuple of (R, G, B) values.
    """
    means, modes, stds, maxs, mins = [], [], [], [], []

    for ch in range(3):
        channel = frame[:, :, ch].ravel()
        means.append(float(np.mean(channel)))
        result = scipy_mode(channel, keepdims=True)
        modes.append(int(result.mode[0]))
        stds.append(float(np.std(channel)))
        maxs.append(int(np.max(channel)))
        mins.append(int(np.min(channel)))

    return {
        'mean': tuple(means),
        'mode': tuple(modes),
        'std':  tuple(stds),
        'max':  tuple(maxs),
        'min':  tuple(mins),
    }


def compute_entropy(frame: np.ndarray) -> tuple:
    """
    Compute Shannon entropy in bits for each channel of an RGB frame.

    Formula: H = -sum(p * log2(p)) for non-zero probabilities,
    where p is the normalised histogram of the channel (256 bins over [0, 256]).

    Args:
        frame: NumPy uint8 array of shape (H, W, 3), channels ordered R, G, B.

    Returns:
        (entropy_r, entropy_g, entropy_b) — each a float in [0.0, 8.0]
    """
    total_pixels = frame.shape[0] * frame.shape[1]
    entropies = []

    for ch in range(3):
        channel = frame[:, :, ch].ravel()
        counts, _ = np.histogram(channel, bins=256, range=(0, 256))
        # Normalise to probabilities
        probs = counts / total_pixels
        # Apply Shannon entropy formula only for non-zero probabilities
        nonzero = probs[probs > 0]
        entropy = -np.sum(nonzero * np.log2(nonzero))
        entropies.append(float(entropy))

    return tuple(entropies)


def linear_transform(frame: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Apply a per-channel linear (affine) transformation to an RGB frame.

    For each channel c: output[c] = clip(alpha * input[c] + beta, 0, 255)

    Channels are processed independently — no channel mixing occurs.

    Args:
        frame: NumPy uint8 array of shape (H, W, 3), channels ordered R, G, B.
        alpha: Gain (multiplicative factor) applied to every channel.
        beta:  Bias (additive offset) applied to every channel.

    Returns:
        Transformed frame, dtype uint8, same shape as input.
    """
    # Use float64 intermediate to avoid overflow/underflow before clamping
    transformed = alpha * frame.astype(np.float64) + beta
    return np.clip(transformed, 0, 255).astype(np.uint8)


import cv2


import mediapipe as mp


# ---------------------------------------------------------------------------
# Module-level MediaPipe face detector initialisation
# ---------------------------------------------------------------------------
try:
    _mp_face_detection = mp.solutions.face_detection
    _mp_drawing = mp.solutions.drawing_utils
    _face_detector = _mp_face_detection.FaceDetection(
        model_selection=0,        # short-range model (< 2 m)
        min_detection_confidence=0.5,
    )
except Exception as _init_exc:
    raise RuntimeError(
        f"nn_inference: failed to initialise MediaPipe FaceDetection — {_init_exc}"
    ) from _init_exc


def nn_inference(frame: np.ndarray) -> np.ndarray:
    """
    Run neural-network face detection on an RGB frame using MediaPipe.

    Detected faces are annotated with a bounding box and key-point dots drawn
    directly onto the frame.  If no face is detected, the original frame is
    returned unmodified.

    The MediaPipe FaceDetection model is loaded once at module import time.
    A RuntimeError is raised (also at import time) if the model fails to
    initialise.

    Args:
        frame: NumPy uint8 array of shape (H, W, 3), channels ordered R, G, B.

    Returns:
        Annotated frame (or original if no detection), dtype uint8, same
        spatial dimensions as input, all values in [0, 255].
    """
    # MediaPipe expects RGB — our frames are already RGB.
    results = _face_detector.process(frame)

    if not results.detections:
        # No face found: return the original frame unmodified.
        return frame

    # Draw annotations onto a copy to avoid modifying the caller's array.
    output = frame.copy()
    for detection in results.detections:
        _mp_drawing.draw_detection(output, detection)

    # Ensure output stays valid uint8 in [0, 255] after drawing.
    return np.clip(output, 0, 255).astype(np.uint8)


def apply_filter(frame: np.ndarray) -> np.ndarray:
    """
    Apply a Gaussian blur filter to each channel of an RGB frame.

    The filter is a 5×5 Gaussian kernel (sigma=1.0) applied independently to
    each channel via cv2.GaussianBlur.  The output is always clamped to
    [0, 255] and cast to uint8.

    Args:
        frame: NumPy uint8 array of shape (H, W, 3), channels ordered R, G, B.

    Returns:
        Filtered frame, dtype uint8, same spatial dimensions as input.
    """
    # GaussianBlur already returns a uint8 array when the input is uint8
    blurred = cv2.GaussianBlur(frame, (5, 5), sigmaX=1.0)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def histogram_equalization(frame: np.ndarray) -> np.ndarray:
    """
    Apply standard CDF-based histogram equalization independently to each
    channel (R, G, B) of an RGB frame.

    Algorithm per channel:
        1. Compute 256-bin histogram.
        2. Compute cumulative distribution function (CDF).
        3. Build LUT: lut = round((cdf - cdf_min) / (total_pixels - cdf_min) * 255)
           Edge case: if total_pixels == cdf_min (constant-zero channel), map to 0.
        4. Apply LUT: equalized_channel = lut[channel]

    Args:
        frame: NumPy uint8 array of shape (H, W, 3), channels ordered R, G, B.

    Returns:
        Equalized frame, dtype uint8, same shape as input.
    """
    total_pixels = frame.shape[0] * frame.shape[1]
    output = np.empty_like(frame)

    for ch in range(3):
        channel = frame[:, :, ch]

        # Step 1: 256-bin histogram
        hist, _ = np.histogram(channel.ravel(), bins=256, range=(0, 256))

        # Step 2: Cumulative distribution function
        cdf = hist.cumsum()

        # Step 3: Build look-up table
        cdf_min = int(cdf.min())
        denominator = total_pixels - cdf_min
        if denominator == 0:
            # Constant channel (all pixels the same value) — map everything to 0
            lut = np.zeros(256, dtype=np.uint8)
        else:
            lut = np.round(
                (cdf - cdf_min) / denominator * 255
            ).astype(np.uint8)

        # Step 4: Apply LUT
        output[:, :, ch] = lut[channel]

    return output
