"""
Method:
1. Extract mean temperature per ROI per frame
2. Bandpass filter to isolate HR or RR frequency
3. FFT to find dominant frequency
4. Convert to BPM

References:
- Garbey et al. (2007)
- Cho et al. (2017)
- Tarmizi et al. (2022)
"""

import numpy as np

from preprocessing.signal_extraction import (
    extract_all_roi_signals,
    interpolate_nan,
)
from preprocessing.peak_extraction import (
    bandpass_filter,
    estimate_frequency_fft,
    estimate_frequency_peaks,
)


class ThermalMeanMethod:
    """
    Mean ROI temperature → FFT → BPM
    """

    def __init__(self, method_config, signal_config):
        self.rois = method_config["rois"]
        self.target = method_config.get(
            "target", "both"
        )
        self.signal_config = signal_config

    def estimate(self, frames, rois_per_frame, fps):
        """
        Run thermal mean estimation
        """
        # extract temperature signals
        roi_signals = extract_all_roi_signals(
            frames, rois_per_frame, self.rois
        )

        method = self.signal_config.get(
            "peak_method", "fft"
        )
        fft_window = self.signal_config.get(
            "fft_window", 512
        )

        # estimate heart rate
        hr_bpm = float("nan")
        hr_roi_results = {}
        hr_signal = None

        if self.target in ("hr", "both"):
            bp = self.signal_config["hr_bandpass"]
            valid_bpms = []
            filtered_signals = []

            for roi_name, signal in (
                roi_signals.items()
            ):
                bpm, filtered = (
                    self._estimate_single_roi(
                        signal, fps, bp,
                        method, fft_window,
                    )
                )
                hr_roi_results[roi_name] = bpm
                if not np.isnan(bpm):
                    valid_bpms.append(bpm)
                if filtered is not None:
                    filtered_signals.append(
                        filtered
                    )

            # heart rate is median of all ROIs
            if valid_bpms:
                hr_bpm = float(
                    np.median(valid_bpms)
                )

            # pick ROI signal that is closest to final value
            if (valid_bpms
                    and filtered_signals):
                best_idx = int(np.argmin([
                    abs(b - hr_bpm)
                    for b in valid_bpms
                ]))
                hr_signal = (
                    filtered_signals[best_idx]
                )

        # estimate respiration rate
        rr_bpm = float("nan")
        rr_roi_results = {}
        rr_signal = None

        if self.target in ("rr", "both"):
            bp = self.signal_config["rr_bandpass"]
            valid_bpms = []
            filtered_signals = []

            for roi_name, signal in (
                roi_signals.items()
            ):
                bpm, filtered = (
                    self._estimate_single_roi(
                        signal, fps, bp,
                        method, fft_window,
                    )
                )
                rr_roi_results[roi_name] = bpm
                if not np.isnan(bpm):
                    valid_bpms.append(bpm)
                if filtered is not None:
                    filtered_signals.append(
                        filtered
                    )

            # median over all ROIs
            if valid_bpms:
                rr_bpm = float(
                    np.median(valid_bpms)
                )

            # pick signal closest to esimated RR
            if (valid_bpms
                    and filtered_signals):
                best_idx = int(np.argmin([
                    abs(b - rr_bpm)
                    for b in valid_bpms
                ]))
                rr_signal = (
                    filtered_signals[best_idx]
                )

        # build result
        return {
            "hr_bpm": hr_bpm,
            "rr_bpm": rr_bpm,
            "roi_results": {
                "hr": hr_roi_results,
                "rr": rr_roi_results,
            },
            "method": "thermal_mean",
            "hr_signal": hr_signal,
            "rr_signal": rr_signal,
        }

    def _estimate_single_roi(self, signal, fps, bp_config, method, fft_window):
        """
        Estimate BPM from one ROI signal
        """
        # fill missing values
        signal_clean = interpolate_nan(signal)

        if np.isnan(signal_clean).all():
            return float("nan"), None

        # check if signal is long enough to be filtered
        min_length = bp_config["order"] * 3 + 1
        if len(signal_clean) < min_length:
            return float("nan"), None

        try:
            filtered = bandpass_filter(
                signal_clean, fps,
                low=bp_config["low"],
                high=bp_config["high"],
                order=bp_config["order"],
            )
        except ValueError:
            return float("nan"), None

        # choose frequency estimation method
        if method == "fft":
            freq_hz = estimate_frequency_fft(
                filtered, fps, fft_window
            )
        elif method == "peak_detection":
            freq_hz = estimate_frequency_peaks(
                filtered, fps
            )
        else:
            freq_hz = estimate_frequency_fft(
                filtered, fps, fft_window
            )

        return freq_hz * 60.0, filtered
