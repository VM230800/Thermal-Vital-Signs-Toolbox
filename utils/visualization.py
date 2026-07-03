"""
utils/visualization.py
======================
Diagnostic plots and optional video clips.
Saved automatically when output.save_plots = true.
Video clips only when output.save_video = true.
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import resample
from preprocessing.roi_extraction import compute_rois


ROI_COLORS = {
    "forehead":    (0, 255, 0),
    "left_cheek":  (255, 0, 0),
    "right_cheek": (255, 0, 0),
    "nose":        (0, 255, 255),
    "philtrum":    (0, 165, 255),
}


def _vital_unit(signal_type):
    """Return correct unit for vital sign type."""
    if signal_type == "hr":
        return "BPM"
    return "BrPM"


def _to_color(frame):
    """
    Convert grayscale frame to colormap (Inferno).
    Returns BGR uint8 image ready for cv2.

    If already 3-channel, just ensures uint8.
    """
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        gray = (
            frame if len(frame.shape) == 2
            else frame[:, :, 0]
        )
        mn, mx = gray.min(), gray.max()
        if mx > mn:
            norm = (
                (gray - mn) / (mx - mn) * 255
            ).astype(np.uint8)
        else:
            norm = np.zeros_like(
                gray, dtype=np.uint8
            )
        return cv2.applyColorMap(
            norm, cv2.COLORMAP_INFERNO
        )
    else:
        if frame.dtype != np.uint8:
            return np.clip(
                frame, 0, 255
            ).astype(np.uint8)
        return frame.copy()


def _draw_overlays(frame, keypoints,
                   frame_idx=None):
    """Draw keypoints + ROI circles on a frame."""
    vis = _to_color(frame)

    for i in range(54):
        if np.isnan(keypoints[i]).any():
            continue
        x = int(keypoints[i, 0])
        y = int(keypoints[i, 1])
        cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(
            vis, str(i), (x + 3, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.25,
            (255, 255, 255), 1,
        )

    if not np.isnan(keypoints).all():
        rois = compute_rois(keypoints)
        for name, (cx, cy, r) in rois.items():
            color = ROI_COLORS.get(
                name, (255, 255, 255)
            )
            cv2.circle(vis, (cx, cy), r, color, 2)
            cv2.putText(
                vis, name, (cx - 20, cy - r - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                color, 1,
            )

    if frame_idx is not None:
        cv2.putText(
            vis, f"Frame {frame_idx}", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1,
        )

    return vis


def _resample_to_match(signal, source_fps,
                       target_fps, target_length):
    """Resample a signal from source_fps to
    target_fps."""
    duration = target_length / target_fps
    n_source = int(duration * source_fps)
    n_source = min(n_source, len(signal))

    if n_source < 2:
        return np.zeros(target_length)

    clipped = signal[:n_source]
    return resample(clipped, target_length)


# ══════════════════════════════════════════════════════════
# 1. ROI Overlay Image (all keypoints)
# ══════════════════════════════════════════════════════════

def save_roi_overlay(frame, keypoints,
                     recording_id, save_dir):
    """Save one annotated frame as SVG."""
    rec_dir = os.path.join(save_dir, recording_id)
    os.makedirs(rec_dir, exist_ok=True)
    vis = _draw_overlays(
        frame, keypoints, frame_idx=0
    )
    path = os.path.join(
        rec_dir,
        f"{recording_id}_roi_overlay.svg",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    fig.savefig(
        path, format="svg",
        bbox_inches="tight", pad_inches=0,
    )
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════
# 1b. Method-specific ROI Overlay
# ══════════════════════════════════════════════════════════

def save_method_roi_overlay(
    frame, keypoints, method_name,
    method_config, recording_id, save_dir,
):
    """
    Save method-specific ROI overlay showing exactly
    which regions each method uses.

    - thermal_mean / ica: circles on ROIs
    - garbey: lines along blood vessels
    """
    rec_dir = os.path.join(save_dir, recording_id)
    os.makedirs(rec_dir, exist_ok=True)

    vis = _to_color(frame)
    h, w = vis.shape[:2]

    method_labels = {
        "thermal_mean": "Thermal Mean",
        "ica": "ICA",
        "garbey": "Garbey (2007)",
    }
    label = method_labels.get(
        method_name, method_name
    )

    cv2.putText(
        vis, f"Method: {label}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2,
    )

    if method_name in ("thermal_mean", "ica"):
        roi_names = method_config.get("rois", [])

        if not np.isnan(keypoints).all():
            rois = compute_rois(keypoints)

            colors = {
                "forehead":    (0, 255, 0),
                "left_cheek":  (255, 100, 100),
                "right_cheek": (100, 100, 255),
                "nose":        (0, 255, 255),
                "philtrum":    (0, 165, 255),
            }

            for name in roi_names:
                if name not in rois:
                    continue

                cx, cy, r = rois[name]
                color = colors.get(
                    name, (255, 255, 255)
                )

                overlay = vis.copy()
                cv2.circle(
                    overlay, (cx, cy), r, color, -1
                )
                cv2.addWeighted(
                    overlay, 0.3, vis, 0.7, 0, vis
                )

                cv2.circle(
                    vis, (cx, cy), r, color, 2
                )

                cv2.putText(
                    vis, name,
                    (cx - 20, cy - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    color, 1,
                )

        cv2.putText(
            vis,
            f"ROIs: {', '.join(roi_names)}",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (255, 255, 255), 1,
        )

    elif method_name == "garbey":
        lines_config = method_config.get(
            "lines", {}
        )

        line_colors = {
            "hr": (0, 0, 255),
            "rr": (255, 0, 0),
        }
        line_labels = {
            "hr": "HR: Temporal Artery",
            "rr": "RR: Nasal Airflow",
        }

        if not np.isnan(keypoints).all():
            y_offset = 50
            n_points = method_config.get(
                "line_points", 10
            )
            lw = method_config.get("line_width", 3)

            for target, line_def in (
                lines_config.items()
            ):
                p1_idx = line_def["p1"]
                p2_idx = line_def["p2"]

                if (
                    np.isnan(
                        keypoints[p1_idx]
                    ).any()
                    or np.isnan(
                        keypoints[p2_idx]
                    ).any()
                ):
                    continue

                p1 = keypoints[p1_idx]
                p2 = keypoints[p2_idx]

                x1 = int(p1[0])
                y1 = int(p1[1])
                x2 = int(p2[0])
                y2 = int(p2[1])

                color = line_colors.get(
                    target, (255, 255, 255)
                )

                cv2.line(
                    vis, (x1, y1), (x2, y2),
                    color, 3,
                )
                cv2.circle(
                    vis, (x1, y1), 5, color, -1
                )
                cv2.circle(
                    vis, (x2, y2), 5, color, -1
                )

                for j in range(n_points):
                    frac = j / max(
                        n_points - 1, 1
                    )
                    px = int(
                        x1 + frac * (x2 - x1)
                    )
                    py = int(
                        y1 + frac * (y2 - y1)
                    )
                    cv2.circle(
                        vis, (px, py), 3,
                        (255, 255, 255), -1,
                    )

                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                direction = np.array(
                    [x2 - x1, y2 - y1],
                    dtype=float,
                )
                length = np.linalg.norm(direction)
                if length > 0:
                    direction /= length
                    perp = np.array(
                        [-direction[1],
                         direction[0]]
                    )
                    pw1 = (
                        int(mid_x - lw * perp[0]),
                        int(mid_y - lw * perp[1]),
                    )
                    pw2 = (
                        int(mid_x + lw * perp[0]),
                        int(mid_y + lw * perp[1]),
                    )
                    cv2.line(
                        vis, pw1, pw2,
                        (255, 255, 0), 1,
                    )

                cv2.putText(
                    vis,
                    line_labels.get(
                        target, target
                    ),
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1,
                )
                y_offset += 20

            cv2.putText(
                vis,
                f"Points: {n_points}, "
                f"Width: {lw}px",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1,
            )

    path = os.path.join(
        rec_dir,
        f"{recording_id}_{method_name}"
        f"_roi_overlay.svg",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    fig.savefig(
        path, format="svg",
        bbox_inches="tight", pad_inches=0,
    )
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════
# 2. Signal Analysis Plot
# ══════════════════════════════════════════════════════════

def save_signal_plot(
    raw_signal, filtered_signal, fps,
    estimated_bpm, ground_truth_bpm,
    target, method_name,
    recording_id, save_dir,
):
    """3-panel plot: raw signal, filtered, FFT."""
    rec_dir = os.path.join(save_dir, recording_id)
    os.makedirs(rec_dir, exist_ok=True)

    unit = _vital_unit(target)

    n = len(raw_signal)
    time = np.arange(n) / fps

    fig, axes = plt.subplots(3, 1, figsize=(10, 7))

    error = abs(estimated_bpm - ground_truth_bpm)
    gt_str = (
        f"{ground_truth_bpm:.1f}"
        if not np.isnan(ground_truth_bpm)
        else "N/A"
    )
    fig.suptitle(
        f"{recording_id} – {method_name}"
        f" – {target.upper()}\n"
        f"Estimated: {estimated_bpm:.1f} {unit}"
        f" | GT: {gt_str} {unit}"
        f" | Error: {error:.1f} {unit}",
        fontsize=11,
    )

    axes[0].plot(
        time, raw_signal[:n],
        color="steelblue", linewidth=0.8,
    )
    axes[0].set_ylabel("Temperature [°C]")
    axes[0].set_title("Raw ROI Signal")
    axes[0].grid(True, alpha=0.3)

    time_filt = (
        np.arange(len(filtered_signal)) / fps
    )
    axes[1].plot(
        time_filt, filtered_signal,
        color="darkorange", linewidth=0.8,
    )
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Bandpass Filtered")
    axes[1].grid(True, alpha=0.3)

    n_fft = len(filtered_signal)
    window = np.hanning(n_fft)
    fft_vals = np.abs(
        np.fft.rfft(filtered_signal * window)
    )
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)

    axes[2].plot(
        freqs, fft_vals,
        color="green", linewidth=0.8,
    )

    peak_freq = estimated_bpm / 60.0
    axes[2].axvline(
        peak_freq, color="red", linestyle="--",
        linewidth=1.2,
        label=(
            f"Estimated: "
            f"{estimated_bpm:.1f} {unit}"
        ),
    )

    if not np.isnan(ground_truth_bpm):
        gt_freq = ground_truth_bpm / 60.0
        axes[2].axvline(
            gt_freq, color="blue", linestyle=":",
            linewidth=1.2,
            label=f"GT: {gt_str} {unit}",
        )

    if target == "hr":
        axes[2].set_xlim(0.5, 4.5)
    else:
        axes[2].set_xlim(0.0, 1.0)

    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("Power")
    axes[2].set_title("FFT Power Spectrum")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(
        rec_dir,
        f"{recording_id}_{method_name}"
        f"_{target}_signal.svg",
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════
# 3. Signal Comparison – Predicted vs Ground Truth
# ══════════════════════════════════════════════════════════

def save_signal_comparison(
    predicted_signal, gt_signal, fps,
    predicted_bpm, gt_bpm,
    signal_type, method_name,
    recording_id, save_dir,
    bandpass=(0.7, 3.5),
    gt_fps=None,
):
    """
    Plot predicted signal vs ground truth signal.
    Dashed lines mark the actual peaks of the
    displayed curves for visual consistency.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    unit = _vital_unit(signal_type)
    type_label = (
        "Heart Rate" if signal_type == "hr"
        else "Respiration"
    )
    error = abs(predicted_bpm - gt_bpm)

    fig.suptitle(
        f"{type_label} Signal Comparison"
        f" – {method_name}\n"
        f"{recording_id}    |    "
        f"Predicted: {predicted_bpm:.1f} {unit}"
        f"    |    "
        f"GT: {gt_bpm:.1f} {unit}    |    "
        f"Error: {error:.1f} {unit}",
        fontsize=13, fontweight="bold",
    )

    n_pred = len(predicted_signal)

    if (gt_fps is not None
            and abs(gt_fps - fps) > 0.1):
        gt_resampled = _resample_to_match(
            gt_signal, gt_fps, fps, n_pred
        )
    else:
        gt_resampled = gt_signal

    def normalise(sig):
        sig = sig - np.nanmean(sig)
        mx = np.nanmax(np.abs(sig))
        if mx > 0:
            sig = sig / mx
        return sig

    pred_clean = np.nan_to_num(
        predicted_signal.copy(), nan=0.0
    )
    gt_clean = np.nan_to_num(
        gt_resampled.copy(), nan=0.0
    )

    pred_norm = normalise(pred_clean)
    gt_norm = normalise(gt_clean)

    n = min(len(pred_norm), len(gt_norm))
    pred_norm = pred_norm[:n]
    gt_norm = gt_norm[:n]
    time_axis = np.arange(n) / fps

    # ══════════════════════════════════════════
    # Upper plot: Time Domain
    # ══════════════════════════════════════════
    ax1 = axes[0]
    ax1.plot(
        time_axis, gt_norm,
        color="red", alpha=0.7, linewidth=1.0,
        label=(
            f"Ground Truth "
            f"({gt_bpm:.1f} {unit})"
        ),
    )
    ax1.plot(
        time_axis, pred_norm,
        color="blue", alpha=0.7, linewidth=1.0,
        label=(
            f"Predicted "
            f"({predicted_bpm:.1f} {unit})"
        ),
    )

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Normalised Amplitude")
    ax1.set_title("Time Domain (normalised)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax1.text(
        0.02, 0.95,
        f"Error: {error:.1f} {unit}",
        transform=ax1.transAxes,
        fontsize=12, fontweight="bold",
        color=(
            "green" if error < 5
            else "orange" if error < 10
            else "red"
        ),
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white", alpha=0.8,
        ),
    )

    # ══════════════════════════════════════════
    # Lower plot: Frequency Domain
    # ══════════════════════════════════════════
    ax2 = axes[1]

    window = np.hanning(n)
    fft_pred = np.abs(
        np.fft.rfft(pred_norm * window)
    )
    fft_gt = np.abs(
        np.fft.rfft(gt_norm * window)
    )
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    bpm_axis = freqs * 60.0

    if signal_type == "hr":
        bpm_range = (40, 180)
    else:
        bpm_range = (5, 35)

    mask = (
        (bpm_axis >= bpm_range[0])
        & (bpm_axis <= bpm_range[1])
    )

    if mask.any():
        bpm_plot = bpm_axis[mask]
        fft_pred_plot = fft_pred[mask]
        fft_gt_plot = fft_gt[mask]

        mx_pred = fft_pred_plot.max()
        mx_gt = fft_gt_plot.max()
        if mx_pred > 0:
            fft_pred_plot = fft_pred_plot / mx_pred
        if mx_gt > 0:
            fft_gt_plot = fft_gt_plot / mx_gt

        # ── Actual peaks of displayed curves ──
        gt_peak_idx = np.argmax(fft_gt_plot)
        pred_peak_idx = np.argmax(fft_pred_plot)
        gt_peak_bpm = float(
            bpm_plot[gt_peak_idx]
        )
        pred_peak_bpm = float(
            bpm_plot[pred_peak_idx]
        )

        ax2.plot(
            bpm_plot, fft_gt_plot,
            color="red", alpha=0.7,
            linewidth=1.5,
            label=(
                f"GT FFT Peak: "
                f"{gt_peak_bpm:.1f} {unit} "
                f"(metadata: {gt_bpm:.1f})"
            ),
        )
        ax2.plot(
            bpm_plot, fft_pred_plot,
            color="blue", alpha=0.7,
            linewidth=1.5,
            label=(
                f"Predicted Peak: "
                f"{pred_peak_bpm:.1f} {unit} "
                f"(method: "
                f"{predicted_bpm:.1f})"
            ),
        )

        ax2.axvline(
            x=gt_peak_bpm, color="red",
            linestyle="--", alpha=0.5,
        )
        ax2.axvline(
            x=pred_peak_bpm, color="blue",
            linestyle="--", alpha=0.5,
        )

        bp_bpm_low = bandpass[0] * 60
        bp_bpm_high = bandpass[1] * 60
        ax2.axvspan(
            bp_bpm_low, bp_bpm_high,
            alpha=0.1, color="green",
            label="Bandpass Range",
        )

    ax2.set_xlabel(f"Frequency [{unit}]")
    ax2.set_ylabel("Normalised Magnitude")
    ax2.set_title("Frequency Domain (FFT)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    rec_dir = os.path.join(save_dir, recording_id)
    os.makedirs(rec_dir, exist_ok=True)

    filename = (
        f"{recording_id}_{method_name}_"
        f"{signal_type}_comparison.svg"
    )
    filepath = os.path.join(rec_dir, filename)
    fig.savefig(
        filepath, dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"    Saved: {filepath}")


# ══════════════════════════════════════════════════════════
# 4. Ground Truth Physiology Plot
# ══════════════════════════════════════════════════════════

def save_gt_physiology_plot(
    physio_signals, predicted_signal,
    fps_video, fps_physio,
    predicted_bpm, gt_bpm,
    signal_type, method_name,
    recording_id, save_dir,
):
    """
    Plot raw ground truth physiology signal alongside
    our predicted thermal signal.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    unit = _vital_unit(signal_type)
    type_label = (
        "Heart Rate" if signal_type == "hr"
        else "Respiration"
    )
    wave_label = (
        "Blood Pressure [mmHg]"
        if signal_type == "hr"
        else "Respiration [Volts]"
    )
    error = abs(predicted_bpm - gt_bpm)

    fig.suptitle(
        f"{type_label}"
        f" – Ground Truth vs Predicted\n"
        f"{recording_id}  |  {method_name}  |  "
        f"Predicted: {predicted_bpm:.1f} {unit}"
        f"  |  GT: {gt_bpm:.1f} {unit}"
        f"  |  Error: {error:.1f} {unit}",
        fontsize=13, fontweight="bold",
    )

    waveform = physio_signals["waveform"]
    rate_bpm = physio_signals["rate_bpm"]

    video_duration = (
        len(predicted_signal) / fps_video
    )
    n_physio = int(video_duration * fps_physio)
    n_physio = min(n_physio, len(waveform))

    waveform_clip = waveform[:n_physio]
    rate_clip = rate_bpm[
        :min(n_physio, len(rate_bpm))
    ]

    time_physio = (
        np.arange(len(waveform_clip)) / fps_physio
    )
    time_rate = (
        np.arange(len(rate_clip)) / fps_physio
    )
    time_video = (
        np.arange(len(predicted_signal)) / fps_video
    )

    ax1 = axes[0]
    ax1.plot(
        time_physio, waveform_clip,
        color="red", linewidth=0.4, alpha=0.8,
    )
    ax1.set_ylabel(wave_label)
    ax1.set_title(
        f"Ground Truth Waveform "
        f"({fps_physio:.0f} Hz sampling)"
    )
    ax1.grid(True, alpha=0.3)

    if video_duration > 5:
        ax1.set_xlim(
            0, min(5, video_duration)
        )
        ax1.text(
            0.98, 0.95,
            f"Showing first 5s of "
            f"{video_duration:.1f}s",
            transform=ax1.transAxes,
            fontsize=9, ha="right", va="top",
            bbox=dict(
                boxstyle="round",
                facecolor="wheat", alpha=0.8,
            ),
        )

    ax2 = axes[1]

    pred_norm = (
        predicted_signal
        - np.mean(predicted_signal)
    )
    mx = np.max(np.abs(pred_norm))
    if mx > 0:
        pred_norm = pred_norm / mx

    ax2.plot(
        time_video, pred_norm,
        color="blue", linewidth=0.8, alpha=0.8,
    )
    ax2.set_ylabel("Normalised Amplitude")
    ax2.set_title(
        f"Predicted Signal – {method_name} "
        f"({fps_video:.0f} Hz sampling)"
    )
    ax2.grid(True, alpha=0.3)

    if video_duration > 5:
        ax2.set_xlim(
            0, min(5, video_duration)
        )

    ax3 = axes[2]

    rate_unit = _vital_unit(signal_type)
    ax3.plot(
        time_rate, rate_clip,
        color="red", linewidth=0.8, alpha=0.7,
        label=(
            f"GT Rate "
            f"(mean: {gt_bpm:.1f} {rate_unit})"
        ),
    )

    ax3.axhline(
        y=predicted_bpm, color="blue",
        linestyle="--", linewidth=1.5,
        label=(
            f"Predicted: "
            f"{predicted_bpm:.1f} {rate_unit}"
        ),
    )

    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel(f"Rate [{rate_unit}]")
    ax3.set_title(
        f"{type_label} Rate over Time"
    )
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    ax3.text(
        0.02, 0.95,
        f"Error: {error:.1f} {rate_unit}",
        transform=ax3.transAxes,
        fontsize=12, fontweight="bold",
        color=(
            "green" if error < 5
            else "orange" if error < 10
            else "red"
        ),
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white", alpha=0.8,
        ),
    )

    plt.tight_layout()

    rec_dir = os.path.join(save_dir, recording_id)
    os.makedirs(rec_dir, exist_ok=True)

    filename = (
        f"{recording_id}_{method_name}_"
        f"{signal_type}_physiology.svg"
    )
    filepath = os.path.join(rec_dir, filename)
    fig.savefig(
        filepath, dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"    Saved: {filepath}")


# ══════════════════════════════════════════════════════════
# 5. Video Clip – optional
# ══════════════════════════════════════════════════════════

def save_roi_video(frames, keypoints, fps,
                   recording_id, save_dir,
                   max_seconds=4):
    """Short video clip with keypoints + ROIs."""
    rec_dir = os.path.join(save_dir, recording_id)
    os.makedirs(rec_dir, exist_ok=True)

    max_frames = int(max_seconds * fps)
    n = min(len(frames), max_frames)

    test_frame = _to_color(frames[0])
    h, w = test_frame.shape[:2]

    path = os.path.join(
        rec_dir,
        f"{recording_id}_roi_video.mp4",
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        path, fourcc, fps, (w, h)
    )

    for i in range(n):
        vis = _draw_overlays(
            frames[i], keypoints[i], frame_idx=i
        )
        out.write(vis)

    out.release()
    print(
        f"  Saved: {path} "
        f"({n} frames, {n / fps:.1f}s)"
    )


# ══════════════════════════════════════════════════════════
# 6. HRV Analysis – IBI + Tachogram + Metrics
# ══════════════════════════════════════════════════════════

def save_hrv_plot(
    signal, fps, ibi_result, hrv_metrics,
    method_name, recording_id, save_dir,
    gt_bpm=None,
):
    """
    3-panel HRV diagnostic plot:
      1. Signal with detected peaks
      2. IBI over time (Tachogram)
      3. HRV metrics summary table
    """
    rec_dir = os.path.join(save_dir, recording_id)
    os.makedirs(rec_dir, exist_ok=True)

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10)
    )

    mean_hr = hrv_metrics['mean_hr_bpm']
    sdnn = hrv_metrics['sdnn_ms']
    rmssd = hrv_metrics['rmssd_ms']
    n_beats = hrv_metrics['n_beats']

    fig.suptitle(
        f"HRV Analysis – {method_name}\n"
        f"{recording_id}    |    "
        f"Mean HR: {mean_hr:.1f} BPM    |    "
        f"Beats detected: {n_beats}",
        fontsize=13, fontweight="bold",
    )

    # ══════════════════════════════════════
    # Panel 1: Signal + detected peaks
    # ══════════════════════════════════════
    ax1 = axes[0]
    time_axis = np.arange(len(signal)) / fps

    ax1.plot(
        time_axis, signal,
        color="steelblue", linewidth=0.8,
        label="Filtered signal",
    )

    peaks = ibi_result["peak_indices"]
    if len(peaks) > 0:
        ax1.plot(
            peaks / fps, signal[peaks],
            "rv", markersize=8, alpha=0.7,
            label=f"Peaks ({n_beats})",
        )

    ax1.set_ylabel("Amplitude")
    ax1.set_title("Cardiac Signal with Peaks")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ══════════════════════════════════════
    # Panel 2: Tachogram (IBI over time)
    # ══════════════════════════════════════
    ax2 = axes[1]
    ibis_ms = ibi_result["ibis_ms"]

    if len(ibis_ms) > 0:
        ibi_times = ibi_result["peak_times"][1:]
        ibi_times = ibi_times[:len(ibis_ms)]

        ax2.plot(
            ibi_times, ibis_ms,
            "o-", color="darkorange",
            markersize=4, linewidth=1.0,
            label="IBI",
        )

        mean_val = np.mean(ibis_ms)
        std_val = np.std(ibis_ms)

        ax2.axhline(
            y=mean_val,
            color="red", linestyle="--",
            linewidth=1.0,
            label=f"Mean: {mean_val:.0f} ms",
        )

        ax2.axhspan(
            mean_val - std_val,
            mean_val + std_val,
            alpha=0.1, color="orange",
            label=f"±1 SD ({std_val:.0f} ms)",
        )

    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("IBI [ms]")
    ax2.set_title(
        "Tachogram (Inter-Beat Intervals)"
    )
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # ══════════════════════════════════════
    # Panel 3: HRV Metrics Table
    # ══════════════════════════════════════
    ax3 = axes[2]
    ax3.axis("off")

    mean_ibi = hrv_metrics['mean_ibi_ms']
    pnn50 = hrv_metrics['pnn50_pct']

    table_data = [
        ["Mean IBI", f"{mean_ibi:.1f} ms"],
        ["Mean HR", f"{mean_hr:.1f} BPM"],
        ["SDNN", f"{sdnn:.1f} ms"],
        ["RMSSD", f"{rmssd:.1f} ms"],
        ["pNN50", f"{pnn50:.1f} %"],
        ["Beats", f"{n_beats}"],
    ]

    if (gt_bpm is not None
            and not np.isnan(gt_bpm)):
        err = abs(mean_hr - gt_bpm)
        table_data.append(
            ["GT HR", f"{gt_bpm:.1f} BPM"]
        )
        table_data.append(
            ["Error", f"{err:.1f} BPM"]
        )

    tbl = ax3.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(0.6, 1.8)
    ax3.set_title(
        "HRV Metrics (Time Domain)", pad=20
    )

    plt.tight_layout()

    filename = (
        f"{recording_id}_{method_name}"
        f"_hrv_analysis.svg"
    )
    filepath = os.path.join(rec_dir, filename)
    fig.savefig(
        filepath, dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"    Saved: {filepath}")
