"""
Based on Garbey et al. (2007)

Method:
1. Define a line along a blood vessel between two keypoints
2. Sample temperature values along this line
3. Average across the line width
4. Per line point: normalise, mirror, compute FFT
5. Average power spectra across all line points
6. Find dominant frequency
7. Subharmonic correction

Reference: 
Garbey M., Sun N., Merla A. & Pavlidis I. (2007). Contract-Free Measurement of Cardiac Pulse Based on the Analysis of Thermal Imagery. IEEE Transactions on Biomedical Engineering, 54(8), 1418-1426.
"""

import warnings
import numpy as np

# Region of Interest: line definitions
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


def _extract_line_values(frames, keypoints, line_def, line_points=10, line_width=3):
    """
    Extracts temperature values along a line for each frame. Samples multiple points between two keypoints and takes pixels around the line for stability
    """
    n_frames, H, W = frames.shape
    result = np.zeros(
        (n_frames, line_points), dtype=np.float32
    )

    for i in range(n_frames):
        kp = keypoints[i]
        # if keypoints are missing reuse the previous frame if possible
        if (np.isnan(kp[line_def["p1"]]).any()
                or np.isnan(
                    kp[line_def["p2"]]
                ).any()):
            if i > 0:
                result[i] = result[i - 1]
            continue

        # start and end point of the line
        p1 = kp[line_def["p1"]]
        p2 = kp[line_def["p2"]]

        # direction vector of the line
        direction = p2 - p1
        length = np.linalg.norm(direction)

        # if points are too close, frame is skipped
        if length < 1e-3:
            if i > 0:
                result[i] = result[i - 1]
            continue

        # normalize direction vector and get perpendicular vector
        direction = direction / length
        perpendicular = np.array(
            [-direction[1], direction[0]]
        )
        
        # sample multiple points along the line
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
                # check if pixel is inside image bounds
                if 0 <= x < W and 0 <= y < H:
                    values.append(
                        frames[i, y, x]
                    )

            if values:
                result[i, j] = np.mean(values)
            elif i > 0:
                result[i, j] = result[i - 1, j]

    return result


def _averaged_power_spectrum(line_values, fps):
    """
    Compute averaged power spectrum across all line points
    """
    n_frames, n_points = line_values.shape
    spectra = []

    for j in range(n_points):
        signal = line_values[:, j]
        # remove mean so DC offset does not mess up FFT
        signal = signal - np.mean(signal)
        # mirror signal for smoothness
        mirrored = np.concatenate(
            [signal, signal[::-1]]
        )
        # compute power spektrum (magnitude squared of FFT)
        spectrum = (
            np.abs(np.fft.rfft(mirrored)) ** 2
        )
        spectra.append(spectrum)

    # average over all line points
    avg_spectrum = np.mean(spectra, axis=0)
    # compute frequency axis for FFT results
    freqs = np.fft.rfftfreq(
        2 * n_frames, d=1.0 / fps
    )

    return freqs, avg_spectrum


def _find_dominant_frequency(freqs, spectrum, low_hz, high_hz):
    """
    Find the frequency with the highest power in the given band
    """
    #only look at frequencies inside the range
    mask = (
        (freqs >= low_hz) & (freqs <= high_hz)
    )
    # check if no frequencies are in range
    if not mask.any():
        warnings.warn(
            f"No FFT bins in "
            f"[{low_hz:.2f}, {high_hz:.2f}] Hz."
            f" Returning band center."
        )
        return (low_hz + high_hz) / 2.0

    band_freqs = freqs[mask]
    band_power = spectrum[mask]

    # pick frequency with maximum power
    return float(
        band_freqs[np.argmax(band_power)]
    )


def _subharmonic_correction(freq_hz, freqs, spectrum, low_hz, high_hz):
    """
    Correct subharmonic detection error
    """
    # check if double frequency is still in valid range
    double_freq = freq_hz * 2.0

    if (double_freq < low_hz
            or double_freq > high_hz):
        return freq_hz

    # find closest frequency component
    idx_original = np.argmin(
        np.abs(freqs - freq_hz)
    )
    idx_double = np.argmin(
        np.abs(freqs - double_freq)
    )

    power_original = spectrum[idx_original]
    power_double = spectrum[idx_double]

    # if double frequency has a similar or higher power we assume that we detected a subharmonic
    if power_double >= 0.5 * power_original:
        return double_freq

    return freq_hz


def _compute_averaged_line_signal(line_values):
    """
    Compute a single 1D signal from the line values by averaging across all line points
    """
    # Average across all line points to get signal (N,)
    signal = np.mean(line_values, axis=1)
    # Remove mean so signal is centered around zero
    signal = signal - np.mean(signal)
    return signal



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
        # if video has color channel, take the first one
        if frames.ndim == 4:
            frames = frames[
                :, :, :, 0
            ].astype(np.float32)

        #  initialize poutputs with NaNs/None
        hr_bpm = float("nan")
        rr_bpm = float("nan")
        hr_signal = None
        rr_signal = None

        # estimate heart rate if requested
        if (self.target in ("hr", "both")
                and "hr" in self.lines):
            hr_bpm, hr_signal = (
                self._estimate_single(
                    frames, keypoints, fps, "hr"
                )
            )

        # estimate respiration rate if requested
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

    def _estimate_single(self, frames, keypoints, fps, target):
        """
        Estimate one vital sign (HR or RR)
        """
        # get line definition
        line_def = self.lines[target]
        # get bandpass settings 
        bp = self.signal_config[
            f"{target}_bandpass"
        ]
        low_hz = bp["low"]
        high_hz = bp["high"]

        # extract line values
        line_values = _extract_line_values(
            frames, keypoints, line_def,
            line_points=self.line_points,
            line_width=self.line_width,
        )

        if np.isnan(line_values).all():
            return float("nan"), None

        # create averaged signal for visualization
        avg_signal = (
            _compute_averaged_line_signal(
                line_values
            )
        )

        # averaged power spectrum
        freqs, spectrum = (
            _averaged_power_spectrum(
                line_values, fps
            )
        )

        # find dominant frequency
        freq_hz = _find_dominant_frequency(
            freqs, spectrum, low_hz, high_hz
        )

        # subharmonic correction
        freq_hz = _subharmonic_correction(
            freq_hz, freqs, spectrum,
            low_hz, high_hz,
        )

        # convert Hz to BPM
        return freq_hz * 60.0, avg_signal
