"""
utils/visualization.py
======================
Diagnostic plots and optional video clips.

Output structure:
  results/
    F001_T1/              ← per recording
      F001_T1_roi_overlay.png
      F001_T1_ica_hr_signal.png
      F001_T1_roi_video.mp4     (optional)
    summary/              ← overall results
      results.csv
      results_table.pdf
      HR_ICA_BlandAltman.pdf
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocessing.roi_extraction import compute_rois


# ══════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════

ROI_COLORS = {
    "forehead":    (0, 255, 0),
    "left_cheek":  (255, 0, 0),
    "right_cheek": (255, 0, 0),
    "nose":        (0, 255, 255),
    "philtrum":    (0, 165, 255),
}


def _get_recording_dir(base_dir, subject, task):
    """Create and return a per-recording subdirectory."""
    recording_dir = os.path.join(base_dir, f"{subject}_{task}")
    os.makedirs(recording_dir, exist_ok=True)
    return recording_dir


def _draw_overlays(frame, keypoints, frame_idx=None):
    """Draw keypoints + ROI circles on a frame."""
    vis = frame.copy()

    for i in range(54):
        if np.isnan(keypoints[i]).any():
            continue
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
        cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                    (255, 255, 255), 1)

    if not np.isnan(keypoints).all():
        rois = compute_rois(keypoints)
        for name, (cx, cy, r) in rois.items():
            color = ROI_COLORS.get(name, (255, 255, 255))
            cv2.circle(vis, (cx, cy), r, color, 2)
            cv2.putText(vis, name, (cx - 20, cy - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        color, 1)

    if frame_idx is not None:
        cv2.putText(vis, f"Frame {frame_idx}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

    return vis


# ══════════════════════════════════════════════════════════
# 1. ROI Overlay Image
# ══════════════════════════════════════════════════════════

def save_roi_overlay(frame, keypoints, subject, task, save_dir):
    """Save one annotated frame as PNG. ~100 KB."""
    rec_dir = _get_recording_dir(save_dir, subject, task)
    vis = _draw_overlays(frame, keypoints, frame_idx=0)
    path = os.path.join(rec_dir, f"{subject}_{task}_roi_overlay.png")
    cv2.imwrite(path, vis)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════
# 2. Signal Analysis Plot
# ══════════════════════════════════════════════════════════

def save_signal_plot(raw_signal, filtered_signal, fps,
                     estimated_bpm, ground_truth_bpm,
                     target, method_name,
                     subject, task, save_dir):
    """3-panel plot: raw → filtered → FFT. ~150 KB."""
    rec_dir = _get_recording_dir(save_dir, subject, task)

    n = len(raw_signal)
    time = np.arange(n) / fps

    fig, axes = plt.subplots(3, 1, figsize=(10, 7))

    # Title with all info
    error = abs(estimated_bpm - ground_truth_bpm)
    gt_str = f"{ground_truth_bpm:.1f}" if not np.isnan(ground_truth_bpm) else "N/A"
    fig.suptitle(
        f"{subject}/{task} – {method_name} – {target.upper()}\n"
        f"Estimated: {estimated_bpm:.1f} BPM | "
        f"Ground Truth: {gt_str} BPM | "
        f"Error: {error:.1f} BPM",
        fontsize=11,
    )

    # Panel 1: Raw
    axes[0].plot(time, raw_signal[:n], color="steelblue",
                 linewidth=0.8)
    axes[0].set_ylabel("Temperature [°C]")
    axes[0].set_title("Raw ROI Signal")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Filtered
    time_filt = np.arange(len(filtered_signal)) / fps
    axes[1].plot(time_filt, filtered_signal, color="darkorange",
                 linewidth=0.8)
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Bandpass Filtered")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: FFT
    n_fft = len(filtered_signal)
    window = np.hanning(n_fft)
    fft_vals = np.abs(np.fft.rfft(filtered_signal * window))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)

    axes[2].plot(freqs, fft_vals, color="green", linewidth=0.8)

    # Estimated peak
    peak_freq = estimated_bpm / 60.0
    axes[2].axvline(peak_freq, color="red", linestyle="--",
                    linewidth=1.2,
                    label=f"Estimated: {estimated_bpm:.1f} BPM")

    # Ground truth
    if not np.isnan(ground_truth_bpm):
        gt_freq = ground_truth_bpm / 60.0
        axes[2].axvline(gt_freq, color="blue", linestyle=":",
                        linewidth=1.2,
                        label=f"GT: {ground_truth_bpm:.1f} BPM")

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
        f"{subject}_{task}_{method_name}_{target}_signal.png"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════
# 3. Video Clip (optional)
# ══════════════════════════════════════════════════════════

def save_roi_video(frames, keypoints, fps, subject, task,
                   save_dir, max_seconds=4):
    """Short video clip with ROIs. ~2-5 MB."""
    rec_dir = _get_recording_dir(save_dir, subject, task)

    max_frames = int(max_seconds * fps)
    n = min(len(frames), max_frames)

    h, w = frames[0].shape[:2]
    path = os.path.join(rec_dir, f"{subject}_{task}_roi_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for i in range(n):
        vis = _draw_overlays(frames[i], keypoints[i], frame_idx=i)
        out.write(vis)

    out.release()
    print(f"  Saved: {path} ({n} frames, {n/fps:.1f}s)")
