# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""
import numpy as np
import cv2

from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, update_histogram
from basics import (
    histogram_figure_numba,
    compute_stats,
    compute_entropy,
    linear_transform,
    histogram_equalization,
    apply_filter,
    nn_inference,
)


# ---------------------------------------------------------------------------
# Histogram figure — sized to fill roughly the top-left quarter of the frame
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def _init_hist_figure(fig_w_px=300, fig_h_px=180, dpi=80):
    """
    Create a histogram figure whose pixel size matches fig_w_px × fig_h_px.
    overlays.py fixes y in [0,3], so we keep that constraint.
    Returns (fig, ax, background, r_plot, g_plot, b_plot).
    """
    fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim([-0.5, 255.5])
    ax.set_ylim([0, 3])
    ax.set_facecolor('#111111')
    fig.patch.set_facecolor('#111111')
    ax.tick_params(colors='white', labelsize=5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)
    x = np.arange(0, 256, 1)
    r_plot = ax.plot(x, np.zeros(256), 'r', animated=True, linewidth=1)[0]
    g_plot = ax.plot(x, np.zeros(256), 'g', animated=True, linewidth=1)[0]
    b_plot = ax.plot(x, np.zeros(256), 'b', animated=True, linewidth=1)[0]
    return fig, ax, background, r_plot, g_plot, b_plot


def _normalise_bars(bars, scale=2.8):
    """Scale histogram bars so peak = scale (fits in overlays.py y-axis [0,3])."""
    peak = float(np.max(bars))
    if peak == 0:
        return bars.astype(np.float64)
    return (bars / peak) * scale


def _overlay_hist_figure(frame, fig):
    """
    Blit the matplotlib figure onto the top-left corner of the frame.
    Dark pixels are treated as transparent (background).
    """
    rgba_buf = fig.canvas.buffer_rgba()
    fw, fh = fig.canvas.get_width_height()
    fig_img = np.frombuffer(rgba_buf, dtype=np.uint8).reshape(fh, fw, 4)[:, :, :3]

    # Clamp to frame size
    h, w = frame.shape[:2]
    fh = min(fh, h)
    fw = min(fw, w)
    fig_img = fig_img[:fh, :fw]

    # Paste: skip near-black background pixels
    region = frame[:fh, :fw].copy()
    mask = np.any(fig_img > 20, axis=2)   # non-background pixels
    region[mask] = fig_img[mask]
    frame[:fh, :fw] = region
    return frame


def _draw_stats(frame, stats, entropy):
    """
    Draw a readable stats panel at the bottom of the frame.

    Layout — two sections stacked vertically at the bottom:
      Section A (3 rows): one row per channel, each showing all 5 metrics
      Section B (1 row):  entropy for R / G / B side by side

    A dark semi-transparent background strip ensures legibility on any content.
    """
    h, w = frame.shape[:2]
    font      = cv2.FONT_HERSHEY_SIMPLEX
    fscale    = 0.45       # slightly larger → more readable
    thick     = 1
    lh        = 18         # line height in pixels
    pad_x     = 6
    pad_y     = 4
    n_rows    = 4          # 3 channel rows + 1 entropy row
    bar_h     = n_rows * lh + pad_y * 2 + 6
    bar_top   = h - bar_h

    # --- semi-transparent dark background ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_top), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # --- channel rows (R, G, B) ---
    ch_colors_bgr = [(60, 60, 255), (60, 210, 60), (255, 100, 60)]  # R G B
    ch_labels     = ('R', 'G', 'B')

    for ch in range(3):
        y = bar_top + pad_y + (ch + 1) * lh
        color = ch_colors_bgr[ch]
        label = ch_labels[ch]
        text = (f"{label}  mean={stats['mean'][ch]:.1f}  "
                f"mode={stats['mode'][ch]}  "
                f"std={stats['std'][ch]:.1f}  "
                f"min={stats['min'][ch]}  "
                f"max={stats['max'][ch]}")
        cv2.putText(frame, text, (pad_x, y),
                    font, fscale, color, thick, cv2.LINE_AA)

    # --- entropy row ---
    y_ent = bar_top + pad_y + 4 * lh
    ent_text = (f"Entropy   R={entropy[0]:.2f}b   "
                f"G={entropy[1]:.2f}b   "
                f"B={entropy[2]:.2f}b")
    cv2.putText(frame, ent_text, (pad_x, y_ent),
                font, fscale, (200, 200, 200), thick, cv2.LINE_AA)

    return frame


def custom_processing(img_source_generator):
    """
    Generator — full processing pipeline applied to every incoming frame.

    Pipeline order:
      1.  compute_stats / compute_entropy  (on raw frame)
      2.  linear_transform  (mild contrast boost)
      3.  histogram_equalization
      4.  apply_filter  (Gaussian blur)
      5.  nn_inference  (MediaPipe face detection)
      6.  histogram_figure_numba  → normalise → update histogram figure
      7.  overlay histogram figure (top-left corner)
      8.  draw stats panel (bottom bar)
      9.  "h" key debounce toggle
      10. yield
    """
    fig, ax, background, r_plot, g_plot, b_plot = _init_hist_figure()

    show_histogram = True
    h_key_cooldown = 0

    for frame in img_source_generator:

        # 1. Stats & entropy on the original frame
        stats = compute_stats(frame)
        ent   = compute_entropy(frame)

        # 2. Mild contrast boost
        frame = linear_transform(frame, alpha=1.2, beta=10.0)

        # 3. Histogram equalization
        frame = histogram_equalization(frame)

        # 4. Gaussian blur filter
        frame = apply_filter(frame)

        # 5. Face detection
        frame = nn_inference(frame)

        # 6. Histogram bars — normalised to [0, 3] for the fixed y-axis
        r_bars, g_bars, b_bars = histogram_figure_numba(frame)

        if show_histogram:
            fig.canvas.restore_region(background)
            r_plot.set_ydata(_normalise_bars(r_bars))
            g_plot.set_ydata(_normalise_bars(g_bars))
            b_plot.set_ydata(_normalise_bars(b_bars))
            ax.draw_artist(r_plot)
            ax.draw_artist(g_plot)
            ax.draw_artist(b_plot)
            fig.canvas.blit(ax.bbox)

            # 7. Paste histogram onto top-left corner of frame
            frame = _overlay_hist_figure(frame, fig)

        # 8. Draw stats panel at the bottom
        frame = _draw_stats(frame, stats, ent)

        # 9. "h" key toggle with 10-frame debounce
        import keyboard
        if keyboard.is_pressed('h'):
            if h_key_cooldown == 0:
                show_histogram = not show_histogram
                h_key_cooldown = 10
        if h_key_cooldown > 0:
            h_key_cooldown -= 1

        yield frame


def main():
    width  = 640
    height = 480
    fps    = 30

    vc = VirtualCamera(fps, width, height)

    try:
        vc.virtual_cam_interaction(
            custom_processing(
                vc.capture_cv_video(0, bgr_to_rgb=True)
            )
        )
    except RuntimeError as e:
        if 'virtual camera' in str(e).lower() or 'backend' in str(e).lower():
            print("\n[INFO] Virtual camera unavailable — opening local preview window instead.")
            print("[INFO] Press 'q' in the preview window to quit.\n")
            import keyboard as _kb
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            def _raw_source():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    yield frame[..., ::-1]  # BGR → RGB

            for processed in custom_processing(_raw_source()):
                # convert RGB back to BGR for cv2.imshow
                cv2.imshow('CV Pipeline Preview', processed[..., ::-1])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            raise


if __name__ == "__main__":
    main()
