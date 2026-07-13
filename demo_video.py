
"""
demo_video.py
=============
Creates a demo video showing thermal frames
alongside live HR and RR estimation graphs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt


# ── Config ──────────────────────────────────────────────
ROOT_DIR = "/Volumes/home/IRData"
SUBJECT = "005"
REC_ID = 0
WINDOW_SEC = 10        # sliding window in seconds
STEP_SEC = 0.5         # step size in seconds
OUTPUT_FILE = "demo_005_rec0.mp4"
OUTPUT_FPS = 30
# ────────────────────────────────────────────────────────


def load_npz(root_dir, subject, rec_id):
    """Load NPZ data."""
    npz_path = os.path.join(
        root_dir, subject,
        f"synchronized_data_{rec_id}.npz")

    print(f"Loading {npz_path}...")
    raw = np.load(npz_path, allow_pickle=True)

    frames = raw["array1"]
    timestamps = raw["array2"]
    pulse = raw["array4"].astype(np.float64)
    resp = raw["array5"].astype(np.float64)
    raw.close()

    # Compute FPS
    median_diff = np.median(np.diff(timestamps))
    if median_diff > 1.0:
        fps = 1000.0 / median_diff
    else:
        fps = 1.0 / median_diff

    print(f"Loaded: {frames.shape[0]} frames, "
          f"FPS={fps:.2f}")
    return frames, pulse, resp, fps


def compute_bpm_windowed(signal, fps,
                         freq_range,
                         window_sec,
                         step_sec):
    """
    Compute BPM over sliding windows.
    Returns arrays of (time_points, bpm_values).
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

        # Bandpass
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

        # FFT
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

        # Time = center of window
        center = (start + end) / 2.0
        times.append(center / fps)

    return np.array(times), np.array(bpms)


def main():
    frames, pulse, resp, fps = load_npz(
        ROOT_DIR, SUBJECT, REC_ID)

    n_frames = frames.shape[0]
    duration = n_frames / fps

    print("Computing HR over sliding windows...")
    hr_times, hr_bpms = compute_bpm_windowed(
        pulse, fps,
        freq_range=(0.7, 3.5),
        window_sec=WINDOW_SEC,
        step_sec=STEP_SEC)

    print("Computing RR over sliding windows...")
    rr_times, rr_bpms = compute_bpm_windowed(
        resp, fps,
        freq_range=(0.1, 0.7),
        window_sec=WINDOW_SEC,
        step_sec=STEP_SEC)

    print(f"HR range: {np.nanmin(hr_bpms):.1f} – "
          f"{np.nanmax(hr_bpms):.1f} BPM")
    print(f"RR range: {np.nanmin(rr_bpms):.1f} – "
          f"{np.nanmax(rr_bpms):.1f} BPM")

    # ── Setup figure ────────────────────────────────
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(2, 2, width_ratios=[1.2, 1],
                  hspace=0.35, wspace=0.3)

    ax_frame = fig.add_subplot(gs[:, 0])
    ax_hr = fig.add_subplot(gs[0, 1])
    ax_rr = fig.add_subplot(gs[1, 1])

    # Frame display
    first_frame = frames[0]
    if first_frame.ndim == 2:
        img = ax_frame.imshow(
            first_frame, cmap="inferno",
            vmin=np.percentile(frames[0], 2),
            vmax=np.percentile(frames[0], 98))
    else:
        img = ax_frame.imshow(first_frame)
    ax_frame.set_axis_off()
    time_text = ax_frame.set_title(
        f"{SUBJECT} / rec_{REC_ID}  —  "
        f"t = 0.00s / {duration:.1f}s",
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

    # ── Animation ───────────────────────────────────
    def update(frame_idx):
        current_time = frame_idx / fps

        # Update thermal frame
        frame = frames[frame_idx]
        img.set_data(frame)
        ax_frame.set_title(
            f"{SUBJECT} / rec_{REC_ID}  —  "
            f"t = {current_time:.1f}s / "
            f"{duration:.1f}s",
            fontsize=12, fontweight="bold")

        # Update HR curve
        hr_mask = hr_times <= current_time
        if hr_mask.any():
            t_vis = hr_times[hr_mask]
            h_vis = hr_bpms[hr_mask]
            hr_line.set_data(t_vis, h_vis)
            hr_dot.set_data(
                [t_vis[-1]], [h_vis[-1]])
            hr_text.set_text(
                f"{h_vis[-1]:.0f} BPM")

        # Update RR curve
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

    print(f"\nRendering {n_frames} frames...")
    anim = animation.FuncAnimation(
        fig, update,
        frames=range(n_frames),
        interval=1000 / fps,
        blit=True)

    writer = animation.FFMpegWriter(
        fps=OUTPUT_FPS,
        bitrate=5000,
        metadata={"title":
                  f"Demo {SUBJECT} rec_{REC_ID}"})

    anim.save(OUTPUT_FILE, writer=writer)
    print(f"\n✅ Video saved: {OUTPUT_FILE}")
    plt.close()


if __name__ == "__main__":
    main()
