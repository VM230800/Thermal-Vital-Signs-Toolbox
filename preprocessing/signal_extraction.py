"""
Extracts temperature time series from ROIs across all frames. No filtering or frequency estimation, only converts spatial ROI data into temporal signals
"""

import numpy as np


def extract_roi_signal(frames, rois_per_frame, roi_name):
    """
    Compute mean temperature of a single ROI across all frames. For each frame, the ROI is a circular patch and the pixel values are averaged
    """
    n_frames = len(frames)
    signal = np.full(n_frames, np.nan, dtype=np.float64)

    for i in range(n_frames):
        rois = rois_per_frame[i]

        # skip frame if no detection or ROI is missing
        if rois is None or roi_name not in rois:
            continue

        cx, cy, r = rois[roi_name]
        frame = frames[i]

        # with a 3-channel only use first channel
        if frame.ndim == 3:
            frame = frame[:, :, 0]

        h, w = frame.shape

        # clamp ROI to image bounds
        y_min = max(0, cy - r)
        y_max = min(h, cy + r)
        x_min = max(0, cx - r)
        x_max = min(w, cx + r)

        # skip if ROI is outside the frame
        if y_min >= y_max or x_min >= x_max:
            continue

        roi_patch = frame[y_min:y_max, x_min:x_max]
        signal[i] = float(np.nanmean(roi_patch))

    return signal


def extract_all_roi_signals(frames, rois_per_frame, roi_names):
    """
    Extract temperature signals for multiple ROIs at once
    """
    signals = {}
    for name in roi_names:
        signals[name] = extract_roi_signal(frames, rois_per_frame, name)
    return signals


def interpolate_nan(signal):
    """
    Fill missing values in a signal using linear interpolation.
    """
    nans = np.isnan(signal)

    # if everything is NaN there is nothing to interpolate
    if nans.all():
        return signal.copy()

    # if no NaN, do nothing
    if not nans.any():
        return signal.copy()

    x = np.arange(len(signal))
    signal_clean = signal.copy()
    # linear interpolation over missing values
    signal_clean[nans] = np.interp(x[nans], x[~nans], signal[~nans])

    return signal_clean
