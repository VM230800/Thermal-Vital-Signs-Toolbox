"""
demo_video.py
=============
Creates a demo video from a thermal .wmv file.
Estimates HR and RR from ROI pixel intensity
and shows live graphs alongside the video.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt


# ── Config ──────────────────────────────────────────────
VIDEO_PATH = ("/Users/valeriamoltschanov/Desktop/"
              "MedTec/Master/4. Semester/PjS/"
              "BP4D+/Thermal/F001/T1.wmv")
WINDOW_SEC = 10
STEP_SEC = 0.5
OUTPUT_FILE = "demo_F001_T1.mp4"
OUTPUT_FPS = 30
# ────────────────────────────────────────────────────────


def load_video(video_path):
    """Load all frames and FPS from video file."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Loading {video_path}")
    print(f"FPS: {fps}, Total frames: {total}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    cap.release()
    frames = np.array(frames)
    print(f"Loaded: {frames.shape}")
    return frames, fps


def extract_roi_signal(frames, roi=None):
    """
    Extract mean pixel intensity from ROI
    for each frame. If no ROI given, use
    center region (forehead approximation).
    """
    h, w = frames.shape[1], frames.shape[2]

    if roi is None:
        # Center-upper region (rough forehead)
        y1 = int(h * 0.15)
        y2 = int(h * 0.40)
        x1 = int(w * 0.30)
        x2 = int(w * 0.70)
    else:
        y1, y2, x1, x2 = roi

    print(f"ROI: y=[{y1}:{y2}], x=[{x1}:{x2}]")

    signal = np.mean(
        frames[:, y1:y2, x1:x2],
        axis=(1, 2)).astype(np.float64)

    return signal


def compute_bpm_windowed(signal, fps,
                         freq_range,
                         window_sec,
                         step_sec):
    """
    Compute BPM over sliding windows.
    """
    window_samples = int(window_sec * fps)
    step_samples = max(1, int(step_sec * fps))
    n = len(signal)

    times = []
    bpms = []

    for start in range(0, n - window_samples + 1,
                       step_samples):
        end = start + window_samples
        segment = signal[start:end].copy()

        segment = segment - np.mean(segment)
        nyq = fps / 2.0
        low = np.clip(
            freq_range[0] / nyq, 0.001, 0.999)
        high = np.clip(
            freq_range[1] / nyq, 0.001, 0.999)

        if low < high and len(segment) > 13:
            b, a = butter(
                4, [low, high], btype="band")
            segment = filtfilt(b, a, segment)

        window = np.hanning(len(segment))
        fft_vals = np.abs(
            np.fft.rfft(segment * window))
        freqs = np.fft.rfftfreq(
            len(segment), d=1.0 / fps)

        mask = ((freqs >= freq_range[0])
                & (freqs <= freq_range[1]))
        if not mask.any():
            bpms.append(float("nan"))
        else:
            fft_masked = fft_vals.copy()
            fft_masked[~mask] = 0
            peak_idx = np.argmax(fft_masked)
            bpms.append(freqs[peak_idx] * 60.0)

        center = (start + end) / 2.0
        times.append(center / fps)

    return np.array(times), np.array(bpms)


def main():
    frames, fps = load_video(VIDEO_PATH)
    n_frames = frames.shape[0]
    duration = n_frames / fps

    # ── Extract signal from ROI ─────────────────
    print("\nExtracting ROI signal...")
    signal = extract_roi_signal(frames)

    # ── Compute HR / RR ─────────────────────────
    print("Computing HR...")
    hr_times, hr_bpms = compute_bpm_windowed(
        signal, fps,
        freq_range=(0.7, 3.5),
        window_sec=WINDOW_SEC,
        step_sec=STEP_SEC)

    print("Computing RR...")
    rr_times, rr_bpms = compute_bpm_windowed(
        signal, fps,
        freq_range=(0.1, 0.7),
        window_sec=WINDOW_SEC,
        step_sec=STEP_SEC)

    print(f"HR range: {np.nanmin(hr_bpms):.1f} – "
          f"{np.nanmax(hr_bpms):.1f} BPM")
    print(f"RR range: {np.nanmin(rr_bpms):.1f} – "
          f"{np.nanmax(rr_bpms):.1f} BPM")

    # ── Setup figure ────────────────────────────
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(2, 2, width_ratios=[1.2, 1],
                  hspace=0.35, wspace=0.3)

    ax_frame = fig.add_subplot(gs[:, 0])
    ax_hr = fig.add_subplot(gs[0, 1])
    ax_rr = fig.add_subplot(gs[1, 1])

    # Frame display
    img = ax_frame.imshow(
        frames[0], cmap="inferno",
        vmin=np.percentile(frames[0], 2),
        vmax=np.percentile(frames[0], 98))
    ax_frame.set_axis_off()
    ax_frame.set_title(
        f"F001 / T1  —  t = 0.0s / {duration:.1f}s",
        fontsize=12, fontweight="bold")

    # HR plot
    hr_line, = ax_hr.plot(
        [], [], color="crimson",
        linewidth=2, label="Heart Rate")
    hr_dot, = ax_hr.plot(
        [], [], "o", color="crimson",
        markersize=8)
    hr_text = ax_hr.text(
        0.98, 0.92, "", transform=ax_hr.transAxes,
        ha="right", va="top", fontsize=16,
        fontweight="bold", color="crimson",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="white", alpha=0.8))
    ax_hr.set_xlim(0, duration)
    ax_hr.set_ylim(
        max(40, np.nanmin(hr_bpms) - 10),
        min(180, np.nanmax(hr_bpms) + 10))
    ax_hr.set_ylabel("BPM", fontsize=11)
    ax_hr.set_title("Heart Rate", fontsize=12,
                    fontweight="bold", color="crimson")
    ax_hr.grid(True, alpha=0.3)
    ax_hr.legend(loc="upper left")

    # RR plot
    rr_line, = ax_rr.plot(
        [], [], color="steelblue",
        linewidth=2, label="Respiratory Rate")
    rr_dot, = ax_rr.plot(
        [], [], "o", color="steelblue",
        markersize=8)
    rr_text = ax_rr.text(
        0.98, 0.92, "", transform=ax_rr.transAxes,
        ha="right", va="top", fontsize=16,
        fontweight="bold", color="steelblue",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="white", alpha=0.8))
    ax_rr.set_xlim(0, duration)
    ax_rr.set_ylim(
        max(5, np.nanmin(rr_bpms) - 3),
        min(40, np.nanmax(rr_bpms) + 3))
    ax_rr.set_xlabel("Time (s)", fontsize=11)
    ax_rr.set_ylabel("BPM", fontsize=11)
    ax_rr.set_title("Respiratory Rate", fontsize=12,
                    fontweight="bold",
                    color="steelblue")
    ax_rr.grid(True, alpha=0.3)
    ax_rr.legend(loc="upper left")

    # ── Animation ───────────────────────────────
    # Skip frames for faster rendering
    skip = max(1, int(fps / OUTPUT_FPS))
    frame_indices = list(range(0, n_frames, skip))

    def update(i):
        frame_idx = frame_indices[i]
        current_time = frame_idx / fps

        # Update thermal frame
        img.set_data(frames[frame_idx])
        ax_frame.set_title(
            f"F001 / T1  —  "
            f"t = {current_time:.1f}s / "
            f"{duration:.1f}s",
            fontsize=12, fontweight="bold")

        # Update HR
        hr_mask = hr_times <= current_time
        if hr_mask.any():
            t_vis = hr_times[hr_mask]
            h_vis = hr_bpms[hr_mask]
            hr_line.set_data(t_vis, h_vis)
            hr_dot.set_data(
                [t_vis[-1]], [h_vis[-1]])
            hr_text.set_text(
                f"{h_vis[-1]:.0f} BPM")

        # Update RR
        rr_mask = rr_times <= current_time
        if rr_mask.any():
            t_vis = rr_times[rr_mask]
            r_vis = rr_bpms[rr_mask]
            rr_line.set_data(t_vis, r_vis)
            rr_dot.set_data(
                [t_vis[-1]], [r_vis[-1]])
            rr_text.set_text(
                f"{r_vis[-1]:.0f} BPM")

        return [img, hr_line, hr_dot, hr_text,
                rr_line, rr_dot, rr_text]

    print(f"\nRendering {len(frame_indices)} frames "
          f"to {OUTPUT_FILE}...")

    anim = animation.FuncAnimation(
        fig, update,
        frames=range(len(frame_indices)),
        interval=1000 / OUTPUT_FPS,
        blit=True)

    writer = animation.FFMpegWriter(
        fps=OUTPUT_FPS, bitrate=5000)

    anim.save(OUTPUT_FILE, writer=writer)
    print(f"\n✅ Video saved: {OUTPUT_FILE}")
    plt.close()


if __name__ == "__main__":
    main()
