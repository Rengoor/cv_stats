import numpy as np
import cv2

from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, update_histogram
from basics import (
    histogram_figure_numba,
    compute_stats_and_entropy_from_hist,
    linear_transform,
    histogram_equalization,
    apply_filter,
    segment_background,
)
from emotion_detection import (
    EmotionDetector,
    build_mood_bg,
    apply_emotion_overlay,
)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def _init_hist_figure(fig_w_px=300, fig_h_px=180, dpi=80):
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
    x = np.arange(0, 256, 1)
    r_plot = ax.plot(x, np.zeros(256), 'r', animated=False, linewidth=1)[0]
    g_plot = ax.plot(x, np.zeros(256), 'g', animated=False, linewidth=1)[0]
    b_plot = ax.plot(x, np.zeros(256), 'b', animated=False, linewidth=1)[0]
    return fig, ax, r_plot, g_plot, b_plot


def _normalise_bars(bars, scale=2.8):
    peak = float(np.max(bars))
    if peak == 0:
        return bars.astype(np.float64)
    return (bars / peak) * scale


def _overlay_hist_figure(frame, fig):
    fig.canvas.draw()

    rgba_buf = fig.canvas.buffer_rgba()
    fw, fh = fig.canvas.get_width_height()
    fig_img = np.frombuffer(rgba_buf, dtype=np.uint8).reshape(fh, fw, 4)[:, :, :3].copy()

    h, w = frame.shape[:2]
    fh = min(fh, h)
    fw = min(fw, w)
    fig_img = fig_img[:fh, :fw]

    alpha = 0.85
    region = frame[:fh, :fw].astype(np.float32)
    blended = (fig_img.astype(np.float32) * alpha + region * (1.0 - alpha))
    frame[:fh, :fw] = np.clip(blended, 0, 255).astype(np.uint8)
    return frame


def _draw_stats(frame, stats, entropy):
    h, w    = frame.shape[:2]
    font    = cv2.FONT_HERSHEY_SIMPLEX
    fscale  = 0.45
    thick   = 1
    lh      = 18
    pad_x   = 6
    pad_y   = 4
    n_rows  = 4
    bar_h   = n_rows * lh + pad_y * 2 + 6
    bar_top = h - bar_h

    canvas = np.zeros_like(frame)

    cv2.rectangle(canvas, (0, bar_top), (w, h), (30, 30, 30), -1)

    ch_colors_bgr = [(255, 100, 60), (60, 210, 60), (60, 60, 255)]
    ch_labels     = ('B', 'G', 'R')
    ch_indices    = (2, 1, 0)

    for row, (label, color, idx) in enumerate(zip(ch_labels, ch_colors_bgr, ch_indices)):
        y    = bar_top + pad_y + (row + 1) * lh
        text = (f"{label}  mean={stats['mean'][idx]:.1f}  "
                f"mode={stats['mode'][idx]}  "
                f"std={stats['std'][idx]:.1f}  "
                f"min={stats['min'][idx]}  "
                f"max={stats['max'][idx]}")
        cv2.putText(canvas, text, (pad_x, y), font, fscale, color, thick, cv2.LINE_AA)

    y_ent    = bar_top + pad_y + 4 * lh
    ent_text = (f"Entropy   B={entropy[2]:.2f}b   "
                f"G={entropy[1]:.2f}b   "
                f"R={entropy[0]:.2f}b")
    cv2.putText(canvas, ent_text, (pad_x, y_ent),
                font, fscale, (200, 200, 200), thick, cv2.LINE_AA)

    canvas = cv2.flip(canvas, 1)

    mask = canvas.any(axis=2)
    alpha = 0.80
    frame[mask] = np.clip(
        canvas[mask].astype(np.float32) * alpha +
        frame[mask].astype(np.float32) * (1 - alpha),
        0, 255
    ).astype(np.uint8)

    return frame


def custom_processing(img_source_generator):
    fig, ax, r_plot, g_plot, b_plot = _init_hist_figure()

    show_histogram       = True
    apply_blur           = True
    apply_bg_replacement = False
    apply_emotion        = False

    h_cd = 0
    b_cd = 0
    s_cd = 0
    e_cd = 0

    import keyboard

    emotion_detector = EmotionDetector(run_every_n_frames=6)
    mood_bg_cache: dict = {}

    _emo_seg       = None
    _emo_seg_ready = False
    try:
        import mediapipe as _mp
        _emo_seg       = _mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
        _emo_seg_ready = True
    except Exception:
        pass

    for frame in img_source_generator:

        h, w         = frame.shape[:2]
        total_pixels = h * w
        r_bars_raw, g_bars_raw, b_bars_raw = histogram_figure_numba(frame)
        stats, ent = compute_stats_and_entropy_from_hist(
            r_bars_raw, g_bars_raw, b_bars_raw, total_pixels
        )

        working_frame = frame.copy()

        if apply_blur:
            working_frame = apply_filter(working_frame)

        if apply_bg_replacement:
            working_frame = segment_background(working_frame)

        working_frame = linear_transform(working_frame, alpha=1.1, beta=5.0)

        if apply_blur:
            r_bars_eq, g_bars_eq, b_bars_eq = histogram_figure_numba(working_frame)
        else:
            r_bars_eq, g_bars_eq, b_bars_eq = r_bars_raw, g_bars_raw, b_bars_raw

        working_frame = histogram_equalization(
            working_frame, r_bars_eq, g_bars_eq, b_bars_eq
        )

        if apply_blur:
            working_frame = apply_filter(working_frame)

        if apply_emotion:
            emotion_result = emotion_detector.analyse(frame)

            seg_mask = None
            if _emo_seg_ready:
                frame.flags.writeable = False
                seg_mask = _emo_seg.process(frame).segmentation_mask
                frame.flags.writeable = True

            mood_bg = None
            if emotion_result is not None:
                from emotion_detection import MOOD_BG_RECIPE
                style = MOOD_BG_RECIPE.get(emotion_result["mood"], "clean")
                if style not in mood_bg_cache:
                    mood_bg_cache[style] = build_mood_bg(h, w, style)
                mood_bg = mood_bg_cache[style]

            working_frame = apply_emotion_overlay(
                working_frame, seg_mask, emotion_result, mood_bg
            )

        if show_histogram:
            r_plot.set_ydata(_normalise_bars(r_bars_raw))
            g_plot.set_ydata(_normalise_bars(g_bars_raw))
            b_plot.set_ydata(_normalise_bars(b_bars_raw))
            working_frame = _overlay_hist_figure(working_frame, fig)

        working_frame = _draw_stats(working_frame, stats, ent)

        if keyboard.is_pressed('h'):
            if h_cd == 0:
                show_histogram = not show_histogram
                print(f"[STATUS] Histogram: {show_histogram}")
                h_cd = 15
        if h_cd > 0:
            h_cd -= 1

        if keyboard.is_pressed('b'):
            if b_cd == 0:
                apply_blur = not apply_blur
                print(f"[STATUS] Blur: {apply_blur}")
                b_cd = 15
        if b_cd > 0:
            b_cd -= 1

        if keyboard.is_pressed('s'):
            if s_cd == 0:
                apply_bg_replacement = not apply_bg_replacement
                print(f"[STATUS] BG Replacement: {apply_bg_replacement}")
                s_cd = 15
        if s_cd > 0:
            s_cd -= 1

        if keyboard.is_pressed('e'):
            if e_cd == 0:
                apply_emotion = not apply_emotion
                print(f"[STATUS] Emotion Detection: {apply_emotion}")
                e_cd = 15
        if e_cd > 0:
            e_cd -= 1

        yield working_frame


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
                    yield frame[..., ::-1]

            for processed in custom_processing(_raw_source()):
                cv2.imshow('CV Pipeline Preview', processed[..., ::-1])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            raise


if __name__ == "__main__":
    main()
