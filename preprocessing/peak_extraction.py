"""
preprocessing/peak_extraction.py
================================
Bandpass filtering and frequency estimation for vital signs.

Input:  1D temperature signal + FPS
Output: Estimated heart rate or respiration rate in BPM

Two methods are supported:
  - "fft":            Dominant frequency via FFT (default)
  - "peak_detection": Count peaks in the filtered signal

Filter parameters are read from the run_config.yaml under "signal:".
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# ──────────────────────────────────────────────────────────────────
# Filtering
# ──────────────────────────────────────────────────────────────────

def bandpass_filter(signal, fs, low, high, order=4):
    """
    Apply a Butterworth bandpass filter.

    Args:
        signal: np.ndarray (N,), input signal (NaN-free)
        fs:     float, sampling frequency in Hz (= video FPS)
        low:    float, lower cutoff frequency in Hz
        high:   float, upper cutoff frequency in Hz
        order:  int, filter order

    Returns:
        np.ndarray (N,), filtered signal
    """
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq

    # Guard: frequencies must be within valid range
    if low_norm <= 0 or high_norm >= 1 or low_norm >= high_norm:
        raise ValueError(
            f"Invalid filter frequencies: low={low}, high={high}, "
            f"fs={fs}. Normalised: [{low_norm:.3f}, {high_norm:.3f}]"
        )

    b, a = butter(order, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, signal)


# ──────────────────────────────────────────────────────────────────
# Frequency estimation
# ──────────────────────────────────────────────────────────────────

def estimate_frequency_fft(signal, fs, fft_window=512):
    """
    Estimate dominant frequency using FFT.

    Takes the strongest frequency component in the signal's
    power spectrum (excluding DC at index 0).

    Args:
        signal:     np.ndarray (N,), filtered signal
        fs:         float, sampling frequency in Hz
        fft_window: int, number of samples for FFT.
                    If the signal is shorter, the full signal is used.

    Returns:
        float, dominant frequency in Hz
    """
    # Use the available signal length, up to fft_window
    n = min(len(signal), fft_window)
    windowed = signal[:n]

    # Apply Hann window to reduce spectral leakage
    window = np.hanning(n)
    windowed = windowed * window

    fft_vals = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Skip DC component (index 0)
    peak_idx = np.argmax(fft_vals[1:]) + 1
    return float(freqs[peak_idx])


def estimate_frequency_peaks(signal, fs):
    """
    Estimate dominant frequency by counting peaks.

    Counts the number of peaks in the filtered signal and
    divides by the signal duration.

    Args:
        signal: np.ndarray (N,), filtered signal
        fs:     float, sampling frequency in Hz

    Returns:
        float, estimated frequency in Hz
    """
    # Minimum distance between peaks: assume at least 0.3s apart
    min_distance = int(0.3 * fs)
    peaks, _ = find_peaks(signal, distance=min_distance)

    if len(peaks) < 2:
        return 0.0

    duration = (peaks[-1] - peaks[0]) / fs
    frequency = (len(peaks) - 1) / duration

    return float(frequency)


# ──────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────

def extract_vital_sign(signal, fs, target, signal_config):
    """
    Complete pipeline: interpolate → filter → estimate → BPM.

    Args:
        signal:        np.ndarray (N,), raw temperature signal from
                       one ROI (may contain NaN)
        fs:            float, sampling frequency in Hz (= video FPS)
        target:        str, "hr" or "rr"
        signal_config: dict, the "signal" section from run_config.yaml
                       Example:
                           {"hr_bandpass": {"low": 0.7, "high": 4.0, "order": 4},
                            "rr_bandpass": {"low": 0.1, "high": 0.5, "order": 4},
                            "fft_window": 512,
                            "peak_method": "fft"}

    Returns:
        float, estimated rate in BPM (or NaN if estimation fails)
    """
    from preprocessing.signal_extraction import interpolate_nan

    # ── Interpolate NaN gaps ──
    signal_clean = interpolate_nan(signal)

    if np.isnan(signal_clean).all():
        return float("nan")

    # ── Select filter parameters ──
    if target == "hr":
        bp = signal_config["hr_bandpass"]
    elif target == "rr":
        bp = signal_config["rr_bandpass"]
    else:
        raise ValueError(f"Unknown target: '{target}'. Use 'hr' or 'rr'.")

    # ── Check minimum signal length ──
    min_length = bp["order"] * 3 + 1
    if len(signal_clean) < min_length:
        return float("nan")

    # ── Bandpass filter ──
    try:
        filtered = bandpass_filter(
            signal_clean, fs,
            low=bp["low"],
            high=bp["high"],
            order=bp["order"],
        )
    except ValueError as e:
        print(f"  Filter error: {e}")
        return float("nan")

    # ── Frequency estimation ──
    method = signal_config.get("peak_method", "fft")
    fft_window = signal_config.get("fft_window", 512)

    if method == "fft":
        freq_hz = estimate_frequency_fft(filtered, fs, fft_window)
    elif method == "peak_detection":
        freq_hz = estimate_frequency_peaks(filtered, fs)
    else:
        raise ValueError(f"Unknown peak_method: '{method}'")

    bpm = freq_hz * 60.0
    return bpm
