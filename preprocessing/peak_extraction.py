"""
Bandpass filtering and frequency estimation
Two methods are supported:
- "fft": Dominant frequency via FFT (default)
- "peak_detection": Count peaks in the filtered signal
Filter parameters are read from the run_config.yaml under "signal:"
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def bandpass_filter(signal, fs, low, high, order=4):
    """
    Apply a Butterworth bandpass filter. Keeps frequencies between low and high cutoff
    """
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq

    # frequencies must be within valid range
    if low_norm <= 0 or high_norm >= 1 or low_norm >= high_norm:
        raise ValueError(
            f"Invalid filter frequencies: low={low}, high={high}, "
            f"fs={fs}. Normalised: [{low_norm:.3f}, {high_norm:.3f}]"
        )

    b, a = butter(order, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, signal)


def estimate_frequency_fft(signal, fs, fft_window=512):
    """
    Estimate dominant frequency using FFT. Takes the strongest frequency component in the signal's power spectrum (excluding DC at index 0)
    """
    # limit signal length for FFT
    n = min(len(signal), fft_window)
    windowed = signal[:n]

    # apply Hann window to reduce edge effects
    window = np.hanning(n)
    windowed = windowed * window

    fft_vals = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # skip DC component (index 0)
    peak_idx = np.argmax(fft_vals[1:]) + 1
    return float(freqs[peak_idx])


def estimate_frequency_peaks(signal, fs):
    """
    Estimate dominant frequency by counting peaks. Counts the number of peaks in the filtered signal and divides by the signal duration
    """
    # minimum distance between peaks: assume at least 0.3s apart
    min_distance = int(0.3 * fs)
    peaks, _ = find_peaks(signal, distance=min_distance)

    if len(peaks) < 2:
        return 0.0

    duration = (peaks[-1] - peaks[0]) / fs
    frequency = (len(peaks) - 1) / duration

    return float(frequency)


def extract_vital_sign(signal, fs, target, signal_config):
    """
    Interpolate → filter → frequency estimation → BPM
    """
    from preprocessing.signal_extraction import interpolate_nan

    # fill missing values
    signal_clean = interpolate_nan(signal)

    if np.isnan(signal_clean).all():
        return float("nan")

    # choose correct bandpass settings
    if target == "hr":
        bp = signal_config["hr_bandpass"]
    elif target == "rr":
        bp = signal_config["rr_bandpass"]
    else:
        raise ValueError(f"Unknown target: '{target}'. Use 'hr' or 'rr'.")

    # check minimum signal length
    min_length = bp["order"] * 3 + 1
    if len(signal_clean) < min_length:
        return float("nan")

    # apply bandpass filter
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

    # choose estimation method
    method = signal_config.get("peak_method", "fft")
    fft_window = signal_config.get("fft_window", 512)

    if method == "fft":
        freq_hz = estimate_frequency_fft(filtered, fs, fft_window)
    elif method == "peak_detection":
        freq_hz = estimate_frequency_peaks(filtered, fs)
    else:
        raise ValueError(f"Unknown peak_method: '{method}'")

    # convert Hz to BPM
    bpm = freq_hz * 60.0
    return bpm
