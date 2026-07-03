"""
methods/garbey.py
=================
Vital sign estimation based on Garbey et al. (2007).

Method:
    1. Define a line along a blood vessel between
       two keypoints
    2. Sample temperature values along this line
    3. Average across the line width
    4. Per line point: normalise, mirror, compute FFT
    5. Average power spectra across all line points
    6. Find dominant frequency
    7. Subharmonic correction

References:
    Garbey, M., Sun, N., Merla, A., & Pavlidis, I.
    (2007). IEEE Trans. Biomed. Eng., 54(8), 1418-1426.
"""

import warnings
import numpy as np


# ─────────────────────────────────────────────────
# Line definitions
# ─────────────────────────────────────────────────

DEFAULT_LINES = {
    "hr": {
        "p1": 17, "p2": 1,
        "name": "temporal_artery_left",
    },
    "rr": {
        "p1": 30, "p2": 49,
        "name": "nasal_airflow",
    },
}


# ─────────────────────────────────────────────────
# Line-based signal extraction
# ─────────────────────────────────────────────────

def _extract_line_values(
    frames, keypoints, line_def,
    line_points=10, line_width=3,
):
    """
    Sample temperature values along a vessel line
    for all frames.
    """
    n_frames, H, W = frames.shape
    result = np.zeros(
        (n_frames, line_points), dtype=np.float32
    )

    for i in range(n_frames):
        kp = keypoints[i]

        if (np.isnan(kp[line_def["p1"]]).any()
                or np.isnan(
                    kp[line_def["p2"]]
                ).any()):
            if i > 0:
                result[i] = result[i - 1]
            continue

        p1 = kp[line_def["p1"]]
        p2 = kp[line_def["p2"]]

        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-3:
            if i > 0:
                result[i] = result[i - 1]
            continue

        direction = direction / length
        perpendicular = np.array(
            [-direction[1], direction[0]]
        )

        for j in range(line_points):
            fraction = j / (line_points - 1)
            center = p1 + fraction * (p2 - p1)

            values = []
            for offset in range(
                -line_width, line_width + 1
            ):
                pixel = (
                    center
                    + offset * perpendicular
                )
                x = int(round(pixel[0]))
                y = int(round(pixel[1]))
                if 0 <= x < W and 0 <= y < H:
                    values.append(
                        frames[i, y, x]
                    )

            if values:
                result[i, j] = np.mean(values)
            elif i > 0:
                result[i, j] = result[i - 1, j]

    return result


# ─────────────────────────────────────────────────
# Spectral analysis
# ─────────────────────────────────────────────────

def _averaged_power_spectrum(line_values, fps):
    """
    Compute averaged power spectrum across all
    line points.
    """
    n_frames, n_points = line_values.shape
    spectra = []

    for j in range(n_points):
        signal = line_values[:, j]
        signal = signal - np.mean(signal)
        mirrored = np.concatenate(
            [signal, signal[::-1]]
        )
        spectrum = (
            np.abs(np.fft.rfft(mirrored)) ** 2
        )
        spectra.append(spectrum)

    avg_spectrum = np.mean(spectra, axis=0)
    freqs = np.fft.rfftfreq(
        2 * n_frames, d=1.0 / fps
    )

    return freqs, avg_spectrum


def _find_dominant_frequency(
    freqs, spectrum, low_hz, high_hz,
):
    """
    Find the frequency with the highest power
    in the given band.
    """
    mask = (
        (freqs >= low_hz) & (freqs <= high_hz)
    )

    if not mask.any():
        warnings.warn(
            f"No FFT bins in "
            f"[{low_hz:.2f}, {high_hz:.2f}] Hz."
            f" Returning band center."
        )
        return (low_hz + high_hz) / 2.0

    band_freqs = freqs[mask]
    band_power = spectrum[mask]

    return float(
        band_freqs[np.argmax(band_power)]
    )


def _subharmonic_correction(
    freq_hz, freqs, spectrum, low_hz, high_hz,
):
    """
    Correct subharmonic detection error.
    """
    double_freq = freq_hz * 2.0

    if (double_freq < low_hz
            or double_freq > high_hz):
        return freq_hz

    idx_original = np.argmin(
        np.abs(freqs - freq_hz)
    )
    idx_double = np.argmin(
        np.abs(freqs - double_freq)
    )

    power_original = spectrum[idx_original]
    power_double = spectrum[idx_double]

    if power_double >= 0.5 * power_original:
        return double_freq

    return freq_hz


# ─────────────────────────────────────────────────
# Averaged line signal for visualisation
# ─────────────────────────────────────────────────

def _compute_averaged_line_signal(line_values):
    """
    Compute a single 1D signal from the line
    values by averaging across all line points.

    This is the signal that the FFT is
    effectively computed on.
    """
    # Average across line points → (N,)
    signal = np.mean(line_values, axis=1)
    # Remove mean (same as in _averaged_power_spectrum)
    signal = signal - np.mean(signal)
    return signal


# ─────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────

class GarbeyMethod:

    def __init__(self, method_config,
                 signal_config):
        self.target = method_config.get(
            "target", "both"
        )
        self.line_points = method_config.get(
            "line_points", 10
        )
        self.line_width = method_config.get(
            "line_width", 3
        )
        self.signal_config = signal_config
        self.lines = method_config.get(
            "lines", DEFAULT_LINES
        )

    def estimate(self, frames, keypoints, fps):
        """
        Run Garbey estimation on one recording.

        Returns:
            dict with hr_bpm, rr_bpm, method,
            hr_signal, rr_signal
        """
        if frames.ndim == 4:
            frames = frames[
                :, :, :, 0
            ].astype(np.float32)

        hr_bpm = float("nan")
        rr_bpm = float("nan")
        hr_signal = None
        rr_signal = None

        if (self.target in ("hr", "both")
                and "hr" in self.lines):
            hr_bpm, hr_signal = (
                self._estimate_single(
                    frames, keypoints, fps, "hr"
                )
            )

        if (self.target in ("rr", "both")
                and "rr" in self.lines):
            rr_bpm, rr_signal = (
                self._estimate_single(
                    frames, keypoints, fps, "rr"
                )
            )

        return {
            "hr_bpm": hr_bpm,
            "rr_bpm": rr_bpm,
            "method": "garbey",
            "hr_signal": hr_signal,
            "rr_signal": rr_signal,
        }

    def _estimate_single(
        self, frames, keypoints, fps, target,
    ):
        """
        Estimate one vital sign (HR or RR).

        Returns:
            tuple: (bpm, signal)
        """
        line_def = self.lines[target]
        bp = self.signal_config[
            f"{target}_bandpass"
        ]
        low_hz = bp["low"]
        high_hz = bp["high"]

        # 1. Extract line values
        line_values = _extract_line_values(
            frames, keypoints, line_def,
            line_points=self.line_points,
            line_width=self.line_width,
        )

        if np.isnan(line_values).all():
            return float("nan"), None

        # 2. Averaged line signal for viz
        avg_signal = (
            _compute_averaged_line_signal(
                line_values
            )
        )

        # 3. Averaged power spectrum
        freqs, spectrum = (
            _averaged_power_spectrum(
                line_values, fps
            )
        )

        # 4. Dominant frequency
        freq_hz = _find_dominant_frequency(
            freqs, spectrum, low_hz, high_hz
        )

        # 5. Subharmonic correction
        freq_hz = _subharmonic_correction(
            freq_hz, freqs, spectrum,
            low_hz, high_hz,
        )

        return freq_hz * 60.0, avg_signal
