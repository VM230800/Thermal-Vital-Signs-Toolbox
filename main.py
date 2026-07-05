"""
Main pipeline for the Thermal Vital Signs Toolbox.

Orchestrates the complete workflow:
    1. Load configuration
    2. Load dataset
    3. YOLO face detection + keypoint extraction
    4. ROI computation from keypoints
    5. Run enabled methods (thermal_mean, ica, garbey)
    6. Visualisation (ROI overlay, signal plots,
       optional video)
    7. Evaluate against ground truth
    8. Combined Bland-Altman plots (all methods
       per dataset)
    9. Save results (CSV, plots, PDF table)

Supports two processing modes:
    - BP4D+: streaming (RAM-friendly, one frame
             at a time)
    - NPZ:   batch load (compressed files,
             streaming too slow)
"""

import argparse
import gc
import os
import time
import warnings

import numpy as np
import pandas as pd
import yaml

# ── Data loading ──
from data.bp4d_loader import BP4DDataset
from data.npz_loader import NPZDataset

# ── Preprocessing ──
from utils.yolo_processing import process_with_yolo
from utils.yolo_processing import (
    process_with_yolo_streaming,
)
from preprocessing.roi_extraction import compute_rois
from preprocessing.hrv_analysis import (
    compute_ibi,
    compute_hrv_metrics,
)

# ── Methods ──
from methods.thermal_mean import ThermalMeanMethod
from methods.ica import ICAMethod
from methods.garbey import GarbeyMethod

# ── Evaluation ──
from evaluation.metrics import evaluate_all_and_plot
from evaluation.results_table import ResultsTable

# ── Visualisation ──
from utils.visualization import (
    save_roi_overlay,
    save_method_roi_overlay,
    save_signal_plot,
    save_signal_comparison,
    save_gt_physiology_plot,
    save_roi_video,
    save_hrv_plot,
)


# --- Load configuration ---

def load_config(config_path):
    """Load main config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


# --- Load dataset --- 

def load_dataset(config, dataset_config):
    """Load the dataset specified in config."""
    dataset_type = config["dataset"]

    if dataset_type == "bp4d":
        dataset = BP4DDataset(
            root_dir=dataset_config["root_dir"],
            subjects=dataset_config.get("subjects"),
            tasks=dataset_config.get("tasks"),
            warmup_seconds=dataset_config.get(
                "warmup_seconds", 0),
            fps=dataset_config.get("fps", 25),
        )
    elif dataset_type == "npz":
        dataset = NPZDataset(
            root_dir=dataset_config["root_dir"],
            subjects=dataset_config.get("subjects"),
            recordings=dataset_config.get(
                "recordings"),
            warmup_seconds=dataset_config.get(
                "warmup_seconds", 10),
            fps=dataset_config.get("fps", 30),
        )
    else:
        raise ValueError(
            f"Unknown dataset: '{dataset_type}'")

    print(
        f"Dataset: {dataset_type}, "
        f"{len(dataset)} samples"
    )
    return dataset


# --- YOLO + ROI computation --- 

def run_yolo_and_rois(frames, config):
    """ Face detection → crop → keypoints → ROIs. All frames in RAM. """
    det = config["detection"]

    cropped_frames, keypoints = process_with_yolo(
        frames=frames,
        model_path=det["model_path"],
        target_size=tuple(det["target_size"]),
        padding=det.get("padding", 50),
    )

    rois_per_frame = _keypoints_to_rois(keypoints)

    n_detected = sum(
        1 for r in rois_per_frame
        if r is not None
    )
    print(
        f"  YOLO: {n_detected}/{len(keypoints)} "
        f"frames with keypoints"
    )

    return cropped_frames, keypoints, rois_per_frame


def run_yolo_and_rois_streaming(
    dataset, idx, config, max_frames=None
):
    """ RAM-friendly: streams frames one-by-one through YOLO. Best for BP4D+ (video files). """
    det = config["detection"]
    processing = config.get("processing", {})
    frame_step = processing.get("frame_step", 1)

    frame_iter = dataset.iter_frames(
        idx,
        max_frames=max_frames,
        frame_step=frame_step,
    )

    cropped_frames, keypoints = (
        process_with_yolo_streaming(
            frame_iterator=frame_iter,
            model_path=det["model_path"],
            target_size=tuple(det["target_size"]),
            padding=det.get("padding", 50),
        )
    )

    rois_per_frame = _keypoints_to_rois(keypoints)

    n_detected = sum(
        1 for r in rois_per_frame
        if r is not None
    )

    effective_fps = dataset.get_metadata(idx)["fps"]
    if frame_step > 1:
        effective_fps /= frame_step
        print(
            f"  Frame step: {frame_step} "
            f"(effective FPS: {effective_fps:.1f})"
        )

    print(
        f"  YOLO: {n_detected}/{len(keypoints)} "
        f"frames with keypoints"
    )

    return cropped_frames, keypoints, rois_per_frame


def _keypoints_to_rois(keypoints):
    """Convert keypoint array to list of ROI dicts."""
    rois_per_frame = []
    for i in range(len(keypoints)):
        kp = keypoints[i]
        if np.isnan(kp).all():
            rois_per_frame.append(None)
        else:
            rois_per_frame.append(compute_rois(kp))
    return rois_per_frame


# --- Run methods --- 

def init_methods(config):
    """Initialise all enabled methods."""
    methods_config = config["methods"]
    signal_config = config["signal"]
    methods = {}

    tm_cfg = methods_config.get("thermal_mean", {})
    if tm_cfg.get("enabled", False):
        methods["thermal_mean"] = ThermalMeanMethod(
            tm_cfg, signal_config
        )

    ica_cfg = methods_config.get("ica", {})
    if ica_cfg.get("enabled", False):
        methods["ica"] = ICAMethod(
            ica_cfg, signal_config
        )

    garb_cfg = methods_config.get("garbey", {})
    if garb_cfg.get("enabled", False):
        methods["garbey"] = GarbeyMethod(
            garb_cfg, signal_config
        )

    print(f"Methods enabled: {list(methods.keys())}")
    return methods


def run_methods(
    methods, cropped_frames, keypoints,
    rois_per_frame, fps
):
    """Run all enabled methods on one sample."""
    results = {}

    for name, method in methods.items():
        try:
            if name == "garbey":
                result = method.estimate(
                    cropped_frames, keypoints, fps
                )
            else:
                result = method.estimate(
                    cropped_frames,
                    rois_per_frame, fps
                )

            results[name] = result
            hr = result['hr_bpm']
            rr = result['rr_bpm']
            print(
                f"  {name}: "
                f"HR={hr:.1f}, RR={rr:.1f} BPM"
            )

        except Exception as e:
            warnings.warn(f"  {name} failed: {e}")
            results[name] = {
                "hr_bpm": float("nan"),
                "rr_bpm": float("nan"),
                "hr_signal": None,
                "rr_signal": None,
                "method": name,
            }

    return results


# --- Visualisation helpers ---

def _resolve_gt_signals(sample, fps):
    """ Determine ground-truth signal sources. Returns (hr_gt_signal, hr_gt_fps, rr_gt_signal, rr_gt_fps). """
    bp_wave = sample.get("bp_waveform", None)
    resp_wave = sample.get("resp_waveform", None)
    physio_fps = sample.get("physio_fps", None)
    gt_pulse = sample.get("pulse_rate", None)
    gt_resp = sample.get("resp_rate", None)

    # HR source
    if bp_wave is not None:
        hr_gt_signal = bp_wave
        hr_gt_fps = physio_fps
    elif gt_pulse is not None:
        hr_gt_signal = gt_pulse
        hr_gt_fps = fps
    else:
        hr_gt_signal = None
        hr_gt_fps = None

    # RR source
    if resp_wave is not None:
        rr_gt_signal = resp_wave
        rr_gt_fps = physio_fps
    elif gt_resp is not None:
        rr_gt_signal = gt_resp
        rr_gt_fps = fps
    else:
        rr_gt_signal = None
        rr_gt_fps = None

    return (hr_gt_signal, hr_gt_fps,
            rr_gt_signal, rr_gt_fps)


def _save_hr_comparison(
    hr_signal, hr_gt_signal, fps, result,
    sample, method_name, recording_id,
    save_dir, signal_config, hr_gt_fps
):
    """Save HR signal comparison plot."""
    hr_bp = signal_config["hr_bandpass"]
    save_signal_comparison(
        predicted_signal=hr_signal,
        gt_signal=hr_gt_signal,
        fps=fps,
        predicted_bpm=result["hr_bpm"],
        gt_bpm=sample.get("hr_bpm", float("nan")),
        signal_type="hr",
        method_name=method_name,
        recording_id=recording_id,
        save_dir=save_dir,
        bandpass=(hr_bp["low"], hr_bp["high"]),
        gt_fps=hr_gt_fps,
    )


def _save_rr_comparison(
    rr_signal, rr_gt_signal, fps, result,
    sample, method_name, recording_id,
    save_dir, signal_config, rr_gt_fps
):
    """Save RR signal comparison plot."""
    rr_bp = signal_config["rr_bandpass"]
    save_signal_comparison(
        predicted_signal=rr_signal,
        gt_signal=rr_gt_signal,
        fps=fps,
        predicted_bpm=result["rr_bpm"],
        gt_bpm=sample.get("rr_bpm", float("nan")),
        signal_type="rr",
        method_name=method_name,
        recording_id=recording_id,
        save_dir=save_dir,
        bandpass=(rr_bp["low"], rr_bp["high"]),
        gt_fps=rr_gt_fps,
    )


def _save_physiology_hr(
    hr_signal, fps, sample, result,
    method_name, recording_id, save_dir
):
    """Save HR physiology plot (BP4D+ only)."""
    bp_wave = sample.get("bp_waveform", None)
    physio_fps = sample.get("physio_fps", None)
    gt_pulse = sample.get("pulse_rate", None)

    if bp_wave is None or not physio_fps:
        return

    rate_bpm = (
        gt_pulse if gt_pulse is not None
        else np.array([])
    )

    save_gt_physiology_plot(
        physio_signals={
            "waveform": bp_wave,
            "rate_bpm": rate_bpm,
        },
        predicted_signal=hr_signal,
        fps_video=fps,
        fps_physio=physio_fps,
        predicted_bpm=result["hr_bpm"],
        gt_bpm=sample.get("hr_bpm", float("nan")),
        signal_type="hr",
        method_name=method_name,
        recording_id=recording_id,
        save_dir=save_dir,
    )


def _save_physiology_rr(
    rr_signal, fps, sample, result,
    method_name, recording_id, save_dir
):
    """Save RR physiology plot (BP4D+ only)."""
    resp_wave = sample.get("resp_waveform", None)
    physio_fps = sample.get("physio_fps", None)
    gt_resp = sample.get("resp_rate", None)

    if resp_wave is None or not physio_fps:
        return

    rate_bpm = (
        gt_resp if gt_resp is not None
        else np.array([])
    )

    save_gt_physiology_plot(
        physio_signals={
            "waveform": resp_wave,
            "rate_bpm": rate_bpm,
        },
        predicted_signal=rr_signal,
        fps_video=fps,
        fps_physio=physio_fps,
        predicted_bpm=result["rr_bpm"],
        gt_bpm=sample.get("rr_bpm", float("nan")),
        signal_type="rr",
        method_name=method_name,
        recording_id=recording_id,
        save_dir=save_dir,
    )


# --- Main visualisation entry point ---

def save_visualisations(
    config, sample, cropped, keypoints,
    rois, sample_results
):
    """ Save diagnostic plots for one recording. Uses hr_signal / rr_signal returned directly by each method – no re-extraction needed. """
    output = config.get("output", {})
    save_dir = output.get("save_dir", "results/")
    signal_config = config["signal"]
    recording_id = sample["recording_id"]
    fps = sample.get("fps", 25.0)

    if not output.get("save_plots", False):
        return

    # ── ROI Overlay ──
    save_roi_overlay(
        cropped[0], keypoints[0],
        recording_id, save_dir
    )

    # ── Method-specific ROI Overlays ──
    for method_name in sample_results.keys():
        method_cfg = config["methods"].get(
            method_name, {}
        )
        try:
            save_method_roi_overlay(
                cropped[0], keypoints[0],
                method_name, method_cfg,
                recording_id, save_dir,
            )
        except Exception as e:
            warnings.warn(
                f"    Method overlay failed "
                f"({method_name}): {e}"
            )

    # ── Optional Video ──
    if output.get("save_video", False):
        max_sec = output.get("video_seconds", 4)
        save_roi_video(
            cropped, keypoints, fps,
            recording_id, save_dir,
            max_seconds=max_sec,
        )

    # ── Resolve GT signals ──
    (hr_gt_signal, hr_gt_fps,
     rr_gt_signal, rr_gt_fps) = (
        _resolve_gt_signals(sample, fps)
    )

    # ── Signal Plots per Method ──
    for method_name, result in sample_results.items():

        hr_bpm = result.get("hr_bpm", float("nan"))
        if np.isnan(hr_bpm):
            continue

        hr_signal = result.get("hr_signal", None)
        rr_signal = result.get("rr_signal", None)

        # ── HR comparison ──
        if (hr_signal is not None
                and hr_gt_signal is not None
                and len(hr_gt_signal) > 10):
            try:
                _save_hr_comparison(
                    hr_signal, hr_gt_signal,
                    fps, result, sample,
                    method_name, recording_id,
                    save_dir, signal_config,
                    hr_gt_fps,
                )
            except Exception as e:
                warnings.warn(
                    f"    HR comparison: {e}"
                )

        # ── RR comparison ──
        rr_bpm = result.get("rr_bpm", float("nan"))
        if (rr_signal is not None
                and rr_gt_signal is not None
                and len(rr_gt_signal) > 10
                and not np.isnan(rr_bpm)):
            try:
                _save_rr_comparison(
                    rr_signal, rr_gt_signal,
                    fps, result, sample,
                    method_name, recording_id,
                    save_dir, signal_config,
                    rr_gt_fps,
                )
            except Exception as e:
                warnings.warn(
                    f"    RR comparison: {e}"
                )

        # ── Physiology HR (BP4D+ only) ──
        if hr_signal is not None:
            try:
                _save_physiology_hr(
                    hr_signal, fps, sample,
                    result, method_name,
                    recording_id, save_dir,
                )
            except Exception as e:
                warnings.warn(
                    f"    HR physiology: {e}"
                )

        # ── Physiology RR (BP4D+ only) ──
        if rr_signal is not None:
            try:
                _save_physiology_rr(
                    rr_signal, fps, sample,
                    result, method_name,
                    recording_id, save_dir,
                )
            except Exception as e:
                warnings.warn(
                    f"    RR physiology: {e}"
                )

        # ── HRV Analysis ──
        if hr_signal is not None:
            try:
                ibi_result = compute_ibi(
                    hr_signal, fps
                )
                hrv_metrics = compute_hrv_metrics(
                    ibi_result
                )
                save_hrv_plot(
                    signal=hr_signal,
                    fps=fps,
                    ibi_result=ibi_result,
                    hrv_metrics=hrv_metrics,
                    method_name=method_name,
                    recording_id=recording_id,
                    save_dir=save_dir,
                    gt_bpm=sample.get(
                        "hr_bpm", None
                    ),
                )
            except Exception as e:
                warnings.warn(
                    f"    HRV analysis: {e}"
                )


# --- Collect results for evaluation ---

def collect_results(method_results, sample,
                    method_name):
    """Convert method output + GT into eval format."""
    return {
        "hr_estimated": method_results.get(
            "hr_bpm", float("nan")
        ),
        "hr_ground_truth": sample.get(
            "hr_bpm", float("nan")
        ),
        "rr_estimated": method_results.get(
            "rr_bpm", float("nan")
        ),
        "rr_ground_truth": sample.get(
            "rr_bpm", float("nan")
        ),
        "subject": sample.get("subject", "?"),
        "task": sample.get("task", "?"),
        "recording_id": sample.get(
            "recording_id", "unknown"
        ),
        "method": method_name,
    }


# --- Main pipeline --- 

def run_pipeline(
    config_path="configs/run_config.yaml",
):
    """Run the full pipeline – multiple datasets."""

    print("=" * 60)
    print("  Thermal Vital Signs Toolbox")
    print("=" * 60)

    # ── Load config ──
    with open(config_path) as f:
        config = yaml.safe_load(f)

    verbose = config.get("output", {}).get(
        "verbose", True
    )

    # ── Init methods ──
    methods = init_methods(config)
    if not methods:
        print("No methods enabled.")
        return

    # ── Init results table ──
    save_dir = config.get("output", {}).get(
        "save_dir", "results/"
    )
    os.makedirs(save_dir, exist_ok=True)
    table = ResultsTable(save_path=save_dir)

    all_results = {name: [] for name in methods}

    # ── Support single or multiple datasets ──
    if "datasets" in config:
        dataset_list = config["datasets"]
    else:
        dataset_list = [{
            "name": config["dataset"],
            "config": config["dataset_config"],
        }]

    total_start = time.time()

    # ── Loop over datasets ──
    for ds_entry in dataset_list:
        ds_name = ds_entry["name"]
        ds_config_path = ds_entry["config"]

        with open(ds_config_path) as f:
            dataset_config = yaml.safe_load(f)

        print(f"\n{'═' * 60}")
        print(f"  Dataset: {ds_name.upper()}")
        print(f"{'═' * 60}")

        # ── Build config for this dataset ──
        ds_config = config.copy()
        ds_config["dataset"] = ds_name
        ds_config["dataset_config"] = ds_config_path

        # ── Load dataset ──
        dataset = load_dataset(
            ds_config, dataset_config
        )

        # ── Processing settings ──
        processing = config.get("processing", {})
        max_frames = processing.get(
            "max_frames", None
        )
        use_streaming = processing.get(
            "streaming", True
        )
        frame_step = processing.get("frame_step", 1)

        # ── Process each sample ──
        for idx in range(len(dataset)):
            meta = dataset.get_metadata(idx)
            recording_id = meta["recording_id"]
            fps = meta["fps"]
            total = meta['total_frames']

            print(f"\n{'─' * 50}")
            print(
                f"  Sample {idx + 1}/"
                f"{len(dataset)}: {recording_id}"
            )
            print(
                f"  Frames: {total}, "
                f"FPS: {fps:.1f}"
            )
            print(f"{'─' * 50}")

            # ── YOLO + ROIs ──
            sample = None
            frames = None

            try:
                if (use_streaming
                        and ds_name != "npz"):
                    cropped, keypoints, rois = (
                        run_yolo_and_rois_streaming(
                            dataset, idx, ds_config,
                            max_frames=max_frames,
                        )
                    )
                else:
                    sample = dataset[idx]
                    frames = sample["frames"]
                    if max_frames:
                        frames = (
                            frames[:max_frames]
                        )
                    cropped, keypoints, rois = (
                        run_yolo_and_rois(
                            frames, ds_config
                        )
                    )

                    # Free raw frames
                    del frames
                    frames = None
                    if "frames" in sample:
                        del sample["frames"]
                    gc.collect()

            except Exception as e:
                warnings.warn(
                    f"  YOLO failed for "
                    f"{recording_id}: {e}"
                )
                if frames is not None:
                    del frames
                if sample is not None:
                    del sample
                gc.collect()
                continue

            # ── Run methods ──
            effective_fps = fps / frame_step
            sample_results = run_methods(
                methods, cropped, keypoints,
                rois, effective_fps,
            )

            # ── Save visualisations ──
            try:
                save_visualisations(
                    ds_config, meta,
                    cropped, keypoints, rois,
                    sample_results,
                )
            except Exception as e:
                warnings.warn(
                    f"  Visualisation failed: {e}"
                )

            # ── Collect for evaluation ──
            for method_name, result in (
                sample_results.items()
            ):
                entry = collect_results(
                    result, meta,
                    method_name,
                )
                entry["dataset"] = ds_name.upper()
                all_results[method_name].append(
                    entry
                )

                if verbose:
                    hr_est = entry['hr_estimated']
                    hr_gt = entry['hr_ground_truth']
                    rr_est = entry['rr_estimated']
                    rr_gt = entry['rr_ground_truth']
                    print(
                        f"    {method_name}: "
                        f"HR {hr_est:.1f}"
                        f" vs {hr_gt:.1f}"
                        f", RR {rr_est:.1f}"
                        f" vs {rr_gt:.1f}"
                    )

            # ── Free RAM ──
            del cropped, keypoints, rois
            del sample_results
            if sample is not None:
                del sample
            gc.collect()

        # ── Free dataset before next one ──
        del dataset
        gc.collect()

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(
        f"  Processing complete in {elapsed:.1f}s"
    )
    print(f"{'=' * 60}")

    # ── Per-Sample CSV ──
    all_rows = []
    for method_name, results in (
        all_results.items()
    ):
        for r in results:
            all_rows.append(r)

    summary_dir = os.path.join(save_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    if all_rows:
        df = pd.DataFrame(all_rows)
        sample_csv = os.path.join(
            summary_dir, "per_sample_results.csv"
        )
        df.to_csv(sample_csv, index=False)
        print(f"Per-sample results: {sample_csv}")

    # ── Evaluate + combined Bland-Altman ──
    for ds_entry in dataset_list:
        ds_name = ds_entry["name"].upper()

        print(f"\n{'─' * 50}")
        print(f"  Evaluating: {ds_name}")
        print(f"{'─' * 50}")

        eval_results = evaluate_all_and_plot(
            all_results, ds_name, summary_dir
        )

        for method_name, result in (
            eval_results.items()
        ):
            table.add(
                ds_name, method_name, "HR",
                result["hr"],
            )
            table.add(
                ds_name, method_name, "RR",
                result["rr"],
            )

    # ── Save results table ──
    table.save_path = summary_dir
    table.print()

    if config.get("output", {}).get(
        "save_csv", True
    ):
        table.save_csv()

    if config.get("output", {}).get(
        "save_plots", True
    ):
        table.save_pdf()

    print(f"\nResults saved to: {save_dir}")
    print("Done.")


# --- CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thermal Vital Signs Toolbox"
    )
    parser.add_argument(
        "--config",
        default="configs/run_config.yaml",
        help="Path to run configuration file",
    )
    args = parser.parse_args()

    run_pipeline(args.config)
