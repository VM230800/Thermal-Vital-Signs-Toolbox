"""
evaluation/metrics.py

Evaluation metrics for HR and RR.

Provides per-method metrics and combined Bland-Altman
plots across all methods for a given dataset.
"""

import numpy as np
from evaluation.bland_altman import (
    BlandAltman,
    combined_difference_plot,
    combined_scatter_plot,
)



# Compute evaluation metrics


def compute_metrics(estimated, ground_truth):
    """
    Compute all evaluation metrics.
    """
    est = np.asarray(estimated, dtype=float)
    gt  = np.asarray(ground_truth, dtype=float)

    valid = ~(np.isnan(est) | np.isnan(gt))
    est, gt = est[valid], gt[valid]
    n = len(est)

    if n == 0:
        return {k: float("nan") for k in
                ("MAE", "MAE_SE", "RMSE", "MAPE",
                 "Pearson", "n")}

    abs_err = np.abs(est - gt)

    return {
        "MAE":     float(np.mean(abs_err)),
        "MAE_SE":  float(
            np.std(abs_err) / np.sqrt(n)),
        "RMSE":    float(
            np.sqrt(np.mean((est - gt) ** 2))),
        "MAPE":    float(
            np.mean(np.abs(
                (est - gt) / (gt + 1e-10)))
            * 100),
        "Pearson": float(
            np.corrcoef(est, gt)[0, 1])
            if n > 1 else float("nan"),
        "n":       n,
    }


def print_metrics(metrics, label=""):
    """
    Print evaluation metrics.
    """
    print(f"  {label}")
    print(f"    MAE     : {metrics['MAE']:.2f} "
          f"± {metrics['MAE_SE']:.2f} BPM")
    print(f"    RMSE    : {metrics['RMSE']:.2f} BPM")
    print(f"    MAPE    : {metrics['MAPE']:.1f}%")
    print(f"    Pearson : {metrics['Pearson']:.3f}")
    print(f"    n       : {metrics['n']}")



# Evaluate a single algorithm (metrics only)


def evaluate_algorithm(results, algo_name,
                       save_dir="results/"):
    """
    Evaluate one algorithm – metrics only.
    """
    hr_est = np.array(
        [r["hr_estimated"] for r in results])
    hr_gt  = np.array(
        [r["hr_ground_truth"] for r in results])
    rr_est = np.array(
        [r["rr_estimated"] for r in results])
    rr_gt  = np.array(
        [r["rr_ground_truth"] for r in results])

    hr_metrics = compute_metrics(hr_est, hr_gt)
    print(f"\n{'=' * 50}")
    print(f"  {algo_name} – Heart Rate")
    print(f"{'=' * 50}")
    print_metrics(hr_metrics, label="HR")

    rr_metrics = compute_metrics(rr_est, rr_gt)
    print(f"\n  {algo_name} – Respiration Rate")
    print(f"{'=' * 50}")
    print_metrics(rr_metrics, label="RR")

    return {
        "algo": algo_name,
        "hr": hr_metrics,
        "rr": rr_metrics,
    }



# Evaluate all methods + combined Bland-Altman


def evaluate_all_and_plot(all_results, dataset_name,
                          save_dir="results/summary"):
    """
    Evaluate all methods for one dataset and create
    combined Bland-Altman plots.

    Parameters
    ----------
    all_results : dict
        {method_name: [list of result dicts], ...}
    dataset_name : str
        e.g. "BP4D" or "NPZ"
    save_dir : str
        Directory for output files

    Returns
    -------
    dict : {method_name: {"hr": metrics,
                          "rr": metrics}}
    """
    eval_results = {}
    hr_data = {}
    rr_data = {}

    for method_name, results in \
            all_results.items():

        ds_results = [
            r for r in results
            if r.get("dataset") == dataset_name
        ]
        if not ds_results:
            continue

        # Per-method metrics
        eval_result = evaluate_algorithm(
            ds_results, method_name, save_dir)
        eval_results[method_name] = eval_result

        # Collect valid pairs for combined plots
        hr_gt = np.array(
            [r["hr_ground_truth"]
             for r in ds_results])
        hr_est = np.array(
            [r["hr_estimated"]
             for r in ds_results])
        rr_gt = np.array(
            [r["rr_ground_truth"]
             for r in ds_results])
        rr_est = np.array(
            [r["rr_estimated"]
             for r in ds_results])

        valid_hr = ~(np.isnan(hr_gt)
                     | np.isnan(hr_est))
        valid_rr = ~(np.isnan(rr_gt)
                     | np.isnan(rr_est))

        if valid_hr.sum() >= 2:
            hr_data[method_name] = (
                hr_gt[valid_hr],
                hr_est[valid_hr],
            )

        if valid_rr.sum() >= 2:
            rr_data[method_name] = (
                rr_gt[valid_rr],
                rr_est[valid_rr],
            )

    # Combined HR plots 
    if hr_data:
        combined_difference_plot(
            methods_data=hr_data,
            the_title=(
                f"Heart Rate – {dataset_name}"),
            save_path=save_dir,
            file_name=(
                f"bland_altman_{dataset_name}"
                f"_HR_combined.pdf"),
            x_label=(
                "Mean of GT and Estimate [BPM]"),
            y_label=(
                "Difference (GT - Est) [BPM]"),
        )
        combined_scatter_plot(
            methods_data=hr_data,
            the_title=(
                f"Heart Rate – {dataset_name}"),
            save_path=save_dir,
            file_name=(
                f"scatter_{dataset_name}"
                f"_HR_combined.pdf"),
            x_label="Ground Truth HR [BPM]",
            y_label="Estimated HR [BPM]",
        )

    # Combined RR plots 
    if rr_data:
        combined_difference_plot(
            methods_data=rr_data,
            the_title=(
                f"Respiration Rate – "
                f"{dataset_name}"),
            save_path=save_dir,
            file_name=(
                f"bland_altman_{dataset_name}"
                f"_RR_combined.pdf"),
            x_label=(
                "Mean of GT and Estimate [BrPM]"),
            y_label=(
                "Difference (GT - Est) [BrPM]"),
        )
        combined_scatter_plot(
            methods_data=rr_data,
            the_title=(
                f"Respiration Rate – "
                f"{dataset_name}"),
            save_path=save_dir,
            file_name=(
                f"scatter_{dataset_name}"
                f"_RR_combined.pdf"),
            x_label="Ground Truth RR [BrPM]",
            y_label="Estimated RR [BrPM]",
        )

    return eval_results
