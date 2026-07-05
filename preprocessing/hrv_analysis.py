"""
preprocessing/hrv_analysis.py
=============================
Inter-Beat Interval (IBI) extraction and
Heart Rate Variability (HRV) analysis.

Input:  Filtered cardiac signal + FPS
Output: IBI series + HRV metrics
"""

import numpy as np
from scipy.signal import find_peaks


def compute_ibi(signal, fps, min_bpm=40,
                max_bpm=180):
    """
    Extract Inter-Beat Intervals from a
    filtered cardiac signal.

    Args:
        signal:  np.ndarray, bandpass-filtered
        fps:     float, sampling rate
        min_bpm: float, minimum expected HR
        max_bpm: float, maximum expected HR

    Returns:
        dict with:
          - peak_indices: indices of detected peaks
          - peak_times:   peak times in seconds
          - ibis:         intervals in seconds
          - ibis_ms:      intervals in milliseconds
    """
    # Min/max distance between peaks
    min_dist = int(fps * 60.0 / max_bpm)
    min_dist = max(min_dist, 1)

    # Find peaks
    peaks, properties = find_peaks(
        signal,
        distance=min_dist,
        height=np.std(signal) * 0.3,
    )

    if len(peaks) < 2:
        return {
            "peak_indices": peaks,
            "peak_times": peaks / fps,
            "ibis": np.array([]),
            "ibis_ms": np.array([]),
        }

    # Calculate IBIs
    peak_times = peaks / fps
    ibis = np.diff(peak_times)

    # Filter physiologically plausible IBIs
    min_ibi = 60.0 / max_bpm
    max_ibi = 60.0 / min_bpm

    valid = (ibis >= min_ibi) & (ibis <= max_ibi)
    ibis_clean = ibis[valid]

    return {
        "peak_indices": peaks,
        "peak_times": peak_times,
        "ibis": ibis_clean,
        "ibis_ms": ibis_clean * 1000.0,
    }


# Alias for backwards compatibility
extract_ibis = compute_ibi


def compute_hrv_metrics(ibi_input):
    """
    Compute standard HRV time-domain metrics.

    Args:
        ibi_input: dict (from compute_ibi) OR
                   np.ndarray of IBIs in ms

    Returns:
        dict with HRV metrics (or NaN if too
        few beats)
    """
    # Accept both dict and raw array
    if isinstance(ibi_input, dict):
        ibis_ms = ibi_input.get(
            "ibis_ms", np.array([]))
    else:
        ibis_ms = np.asarray(ibi_input)

    if len(ibis_ms) < 3:
        return {
            "mean_ibi_ms": float("nan"),
            "sdnn_ms": float("nan"),
            "rmssd_ms": float("nan"),
            "pnn50_pct": float("nan"),
            "mean_hr_bpm": float("nan"),
            "n_beats": len(ibis_ms),
        }

    # Successive differences
    diffs = np.diff(ibis_ms)

    # Metrics
    mean_ibi = float(np.mean(ibis_ms))
    sdnn = float(np.std(ibis_ms, ddof=1))
    rmssd = float(
        np.sqrt(np.mean(diffs ** 2))
    )
    pnn50 = float(
        np.sum(np.abs(diffs) > 50.0)
        / len(diffs) * 100.0
    )
    mean_hr = 60000.0 / mean_ibi

    return {
        "mean_ibi_ms": mean_ibi,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "pnn50_pct": pnn50,
        "mean_hr_bpm": mean_hr,
        "n_beats": len(ibis_ms) + 1,
    }
