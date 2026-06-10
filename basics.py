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

def compute_stats_and_entropy_from_hist(r_bars, g_bars, b_bars, total_pixels):
    """
    Blazing fast metrics computation using already calculated histogram arrays.
    """
    pixel_values = np.arange(256)
    
    means = []
    modes = []
    stds = []
    entropies = []
    maxs = []
    mins = []
    
    channels_bars = [r_bars, g_bars, b_bars]
    
    for bars in channels_bars:
        # 1. Mode (Fastest possible way)
        modes.append(int(np.argmax(bars)))
        
        # 2. Mean
        mean_val = np.sum(pixel_values * bars) / total_pixels
        means.append(float(mean_val))
        
        # 3. Standard Deviation 
        variance = np.sum(bars * (pixel_values - mean_val) ** 2) / total_pixels
        stds.append(float(np.sqrt(variance)))
        
        # 4. Min / Max
        active_indices = np.where(bars > 0)[0]
        mins.append(int(active_indices[0]) if len(active_indices) > 0 else 0)
        maxs.append(int(active_indices[-1]) if len(active_indices) > 0 else 0)
        
        # 5. Entropy (Shannon)
        probs = bars / total_pixels
        nonzero = probs[probs > 0]
        entropy_val = -np.sum(nonzero * np.log2(nonzero))
        entropies.append(float(entropy_val))
        
    stats = {
        'mean': tuple(means),
        'mode': tuple(modes),
        'std':  tuple(stds),
        'max':  tuple(maxs),
        'min':  tuple(mins),
    }
    
    return stats, tuple(entropies)


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


'''import mediapipe as mp


# ---------------------------------------------------------------------------
# Module-level MediaPipe face detector initialisation
# ---------------------------------------------------------------------------
try:
    # Explicitly pull from the underlying python subdirectory to bypass wrapper layer blocks
    import mediapipe.python.solutions.face_detection as mp_face_detection
    import mediapipe.python.solutions.drawing_utils as mp_drawing_utils
    
    _mp_face_detection = mp_face_detection
    _mp_drawing = mp_drawing_utils
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
    return np.clip(output, 0, 255).astype(np.uint8)'''


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
    blurred = cv2.GaussianBlur(frame, (21, 21), sigmaX=1.0)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def histogram_equalization(frame: np.ndarray, r_bars, g_bars, b_bars) -> np.ndarray:
    """
    Ultra-optimized histogram equalization that reuses pre-calculated Numba bars.
    """
    total_pixels = frame.shape[0] * frame.shape[1]
    output = np.empty_like(frame)
    
    # Bundle the pre-calculated bars together
    channels_bars = [r_bars, g_bars, b_bars]

    for ch in range(3):
        channel = frame[:, :, ch]
        hist = channels_bars[ch]  # Use the bars we already computed!

        # Step 2: Cumulative distribution function
        cdf = hist.cumsum()

        # Step 3: Build look-up table
        cdf_min = int(cdf.min())
        denominator = total_pixels - cdf_min
        if denominator == 0:
            lut = np.zeros(256, dtype=np.uint8)
        else:
            lut = np.round((cdf - cdf_min) / denominator * 255).astype(np.uint8)

        # Step 4: Apply LUT
        output[:, :, ch] = lut[channel]

    return output