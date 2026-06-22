
"""
methods/ica.py
==============
ICA-based vital sign estimation from thermal video.

Uses Independent Component Analysis to separate the mixed
temperature signals from multiple facial ROIs into independent
source signals. The source with the strongest spectral peak
in the expected frequency band is selected as the vital sign.

Why ICA works:
    Multiple ROIs capture the same blood pulse signal, but with
    different mixing weights (due to varying skin thickness,
    vessel density, etc.). ICA recovers the original pulse
    signal by finding statistically independent components.

References:
    - Poh et al. (2010): "Non-contact, automated cardiac pulse
      measurements using video imaging and blind source separation"
    - Adapted from rPPG (visible light) to thermal imaging
"""

import warnings
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.decomposition import FastICA

from preprocessing.signal_extraction import (
    extract_all_roi_signals,
    interpolate_nan,
)


# ─────────────────────────────────────────────────────────────────
# ICA-specific preprocessing
# ─────────────────────────────────────────────────────────────────

def _detrend(signal, fps):
    """
    Remove slow drift using Savitzky-Golay filter.

    Thermal signals have a slow baseline drift caused by
    environmental temperature changes and camera drift.
    This must be removed BEFORE ICA, otherwise the drift
    dominates the independent components.

    Args:
        signal: np.ndarray (N,), temperature time series
        fps:    float, sampling rate

    Returns:
        np.ndarray (N,), detrended signal
    """
    window = int(2.0 * fps)
    if window % 2 == 0:
        window += 1
    if len(signal) <= window:
        return signal - np.mean(signal)

    trend = savgol_filter(signal, window, polyorder=2)
    return signal - trend


def _preprocess_for_ica(signals, fps, low_hz, high_hz):
    """
    Prepare multi-ROI signals for ICA decomposition.

    Pipeline per channel:
        1. Interpolate NaN gaps
        2. Remove slow drift (Savitzky-Golay)
        3. Bandpass filter to target frequency range
        4. Z-score normalisation (zero mean, unit variance)

    Args:
        signals: dict, roi_name → np.ndarray (N,)
        fps:     float, sampling rate
        low_hz:  float, lower bandpass cutoff
        high_hz: float, upper bandpass cutoff

    Returns:
        np.ndarray (N, C), preprocessed signals ready for ICA
        list of str, ROI names in column order
    """
    nyq = fps / 2.0
    low_norm = np.clip(low_hz / nyq, 0.001, 0.999)
    high_norm = np.clip(high_hz / nyq, 0.001, 0.999)

    roi_names = []
    channels = []

    for name, raw_signal in signals.items():
        # 1. Interpolate NaN
        s = interpolate_nan(raw_signal)
        if np.isnan(s).all():
            continue

        # 2. Detrend
        s = _detrend(s, fps)

        # 3. Bandpass
        if low_norm < high_norm:
            b, a = butter(4, [low_norm, high_norm], btype="band")
            s = filtfilt(b, a, s)

        # 4. Normalise
        std = s.std()
        if std > 1e-10:
            s = (s - s.mean()) / std
        else:
            continue

        roi_names.append(name)
        channels.append(s)

    if len(channels) == 0:
        return None, []

    return np.column_stack(channels), roi_names


# ─────────────────────────────────────────────────────────────────
# ICA decomposition
# ─────────────────────────────────────────────────────────────────

def _run_ica(signals, n_components=None):
    """
    Apply FastICA to multi-channel signal matrix.

    Args:
        signals:      np.ndarray (N, C), preprocessed ROI signals
        n_components: int or None (default: same as C)

    Returns:
        np.ndarray (N, K), independent components
    """
    if n_components is None:
        n_components = signals.shape[1]

    n_components = min(n_components, signals.shape[1])

    ica = FastICA(
        n_components=n_components,
        max_iter=1500,
        tol=1e-4,
        random_state=42,
    )

    try:
        return ica.fit_transform(signals)
    except Exception as e:
        warnings.warn(f"ICA failed: {e}. Returning original signals.")
        return signals


def _select_best_component(components, fps, low_hz, high_hz):
    """
    Select the ICA component with the strongest spectral peak
    in the target frequency band.

    Uses a simple SNR metric: peak magnitude in band divided by
    total band energy. The component with the highest SNR is
    most likely to contain the vital sign signal.

    Args:
        components: np.ndarray (N, K), ICA output
        fps:        float, sampling rate
        low_hz:     float, lower frequency bound
        high_hz:    float, upper frequency bound

    Returns:
        best_freq:  float, dominant frequency in Hz
        best_idx:   int, index of the best component
        snr_scores: list of float, SNR per component
    """
    N, K = components.shape

    # Zero-pad to next power of 2 for cleaner FFT
    nfft = 1
    while nfft < N:
        nfft <<= 1
    nfft *= 2

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fps)
    mask = (freqs >= low_hz) & (freqs <= high_hz)

    if not mask.any():
        return 0.0, 0, [0.0] * K

    best_snr = -1.0
    best_freq = 0.0
    best_idx = 0
    snr_scores = []

    for k in range(K):
        mag = np.abs(np.fft.rfft(components[:, k], n=nfft))
        band_mag = mag[mask]
        snr = float(band_mag.max() / (band_mag.sum() + 1e-10))
        snr_scores.append(snr)

        if snr > best_snr:
            best_snr = snr
            best_idx = k
            best_freq = float(freqs[mask][np.argmax(band_mag)])

    return best_freq, best_idx, snr_scores


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

class ICAMethod:
    """
    ICA-based vital sign estimator.

    Usage:
        method = ICAMethod(config["methods"]["ica"], config["signal"])
        result = method.estimate(frames, rois_per_frame, fps)
    """

    def __init__(self, method_config, signal_config):
        """
        Args:
            method_config: dict from run_config.yaml, e.g.
                {"enabled": True, "target": "both",
                 "rois": ["forehead", "left_cheek", ...],
                 "n_components": null}
            signal_config: dict from run_config.yaml "signal" section
        """
        self.rois = method_config["rois"]
        self.target = method_config.get("target", "both")
        self.n_components = method_config.get("n_components", None)
        self.signal_config = signal_config

    def estimate(self, frames, rois_per_frame, fps):
        """
        Run ICA-based estimation on one recording.

        Args:
            frames:         np.ndarray (N, H, W) or (N, H, W, 3)
            rois_per_frame: list of dict (from roi_extraction)
            fps:            float

        Returns:
            dict with keys:
                - hr_bpm:   float (or NaN if target is "rr")
                - rr_bpm:   float (or NaN if target is "hr")
                - hr_component_idx: int
                - rr_component_idx: int
                - method:   str, "ica"
        """
        # ── 1. Extract temperature signals per ROI ──
        roi_signals = extract_all_roi_signals(
            frames, rois_per_frame, self.rois
        )

        # Check we have enough signals for ICA
        valid_count = sum(
            1 for s in roi_signals.values()
            if not np.isnan(s).all()
        )
        if valid_count < 2:
            warnings.warn("ICA needs at least 2 valid ROI signals.")
            return self._empty_result()

        # ── 2. Estimate HR ──
        hr_bpm = float("nan")
        hr_idx = -1

        if self.target in ("hr", "both"):
            hr_bp = self.signal_config["hr_bandpass"]
            matrix_hr, names_hr = _preprocess_for_ica(
                roi_signals, fps, hr_bp["low"], hr_bp["high"]
            )

            if matrix_hr is not None and matrix_hr.shape[1] >= 2:
                components_hr = _run_ica(matrix_hr, self.n_components)
                hr_freq, hr_idx, _ = _select_best_component(
                    components_hr, fps, hr_bp["low"], hr_bp["high"]
                )
                hr_bpm = hr_freq * 60.0

        # ── 3. Estimate RR ──
        rr_bpm = float("nan")
        rr_idx = -1

        if self.target in ("rr", "both"):
            rr_bp = self.signal_config["rr_bandpass"]
            matrix_rr, names_rr = _preprocess_for_ica(
                roi_signals, fps, rr_bp["low"], rr_bp["high"]
            )

            if matrix_rr is not None and matrix_rr.shape[1] >= 2:
                components_rr = _run_ica(matrix_rr, self.n_components)
                rr_freq, rr_idx, _ = _select_best_component(
                    components_rr, fps, rr_bp["low"], rr_bp["high"]
                )
                rr_bpm = rr_freq * 60.0

        return {
            "hr_bpm":           hr_bpm,
            "rr_bpm":           rr_bpm,
            "hr_component_idx": hr_idx,
            "rr_component_idx": rr_idx,
            "method":           "ica",
        }

    @staticmethod
    def _empty_result():
        return {
            "hr_bpm":           float("nan"),
            "rr_bpm":           float("nan"),
            "hr_component_idx": -1,
            "rr_component_idx": -1,
            "method":           "ica",
        }
