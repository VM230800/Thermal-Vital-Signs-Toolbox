"""
evaluation/bland_altman.py

Bland-Altman statistics and plots for the Thermal Vital Signs Toolbox.

Computes agreement metrics between a ground-truth reference signal
(e.g. contact ECG or respiratory belt) and a contactless estimate
derived from thermal video.

Four plot types are provided:
    scatter_plot              -- single method scatter
    difference_plot           -- single method Bland-Altman
    combined_scatter_plot     -- all methods in one scatter
    combined_difference_plot  -- all methods in one Bland-Altman

All plots are saved as PDF files (vector format, publication-ready).

Usage
-----
    from evaluation.bland_altman import BlandAltman, combined_difference_plot

    ba = BlandAltman(
        gold_std    = [72.1, 68.4, 75.0, ...],
        new_measure = [69.3, 70.1, 74.2, ...],
        save_path   = "results/bland_altman",
    )
    ba.print_stats()
    ba.difference_plot(the_title="Heart Rate -- ICA method on BP4D+")
    ba.scatter_plot(the_title="Heart Rate -- ICA method on BP4D+")

    # Combined plot for multiple methods:
    combined_difference_plot(
        methods_data = {
            "thermal_mean": (gt_values, est_values),
            "ica":          (gt_values, est_values),
            "garbey":       (gt_values, est_values),
        },
        the_title = "Heart Rate – BP4D+",
        save_path = "results/summary",
    )
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


# Method colours used across all combined plots
METHOD_COLORS = {
    "thermal_mean": "#2196F3",
    "ica":          "#FF9800",
    "garbey":       "#4CAF50",
}

METHOD_MARKERS = {
    "thermal_mean": "o",
    "ica":          "s",
    "garbey":       "D",
}


class BlandAltman:
    """
    Computes Bland-Altman agreement statistics and generates plots.

    Parameters
    ----------
    gold_std    : array-like -- ground-truth values (contact sensor)
    new_measure : array-like -- toolbox estimates (thermal video)
    save_path   : str        -- directory where plots are saved
                               (created automatically if it does not exist)
    averaged    : bool       -- set True when each data point is already
                               a subject-level mean; adjusts the CI95
                               formula accordingly
    """

    def __init__(
        self,
        gold_std,
        new_measure,
        save_path: str = "results/bland_altman",
        averaged: bool = False,
    ):
        self.gold_std    = self._to_series(gold_std,    "gold_std")
        self.new_measure = self._to_series(new_measure, "new_measure")

        if len(self.gold_std) != len(self.new_measure):
            raise ValueError(
                f"gold_std and new_measure must have the same length "
                f"({len(self.gold_std)} vs {len(self.new_measure)})."
            )

        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        # Core statistics
        diffs = self.gold_std - self.new_measure

        self.mean_error              = float(diffs.mean())
        self.std_error               = float(diffs.std())
        self.mean_absolute_error     = float(diffs.abs().mean())
        self.mean_squared_error      = float((diffs ** 2).mean())
        self.root_mean_squared_error = float(np.sqrt(self.mean_squared_error))
        self.correlation             = float(
            np.corrcoef(self.gold_std, self.new_measure)[0, 1]
        )

        # 95% limits of agreement
        if averaged:
            effective_std = np.sqrt(2.0) * self.std_error
        else:
            effective_std = self.std_error

        self.CI95 = [
            self.mean_error + 1.96 * effective_std,
            self.mean_error - 1.96 * effective_std,
        ]

    
    # Statistics output
    

    def print_stats(self, round_amount: int = 4) -> None:
        """
        Print all computed metrics to stdout.
        """
        r = round_amount
        print(f"Mean error               = {round(self.mean_error,              r)}")
        print(f"Mean absolute error      = {round(self.mean_absolute_error,     r)}")
        print(f"Mean squared error       = {round(self.mean_squared_error,      r)}")
        print(f"Root mean squared error  = {round(self.root_mean_squared_error, r)}")
        print(f"Standard deviation error = {round(self.std_error,               r)}")
        print(f"Correlation              = {round(self.correlation,             r)}")
        print(f"+95% Limit of Agreement  = {round(self.CI95[0],                 r)}")
        print(f"-95% Limit of Agreement  = {round(self.CI95[1],                 r)}")

    def return_stats(self) -> dict:
        """
        Return all metrics as a plain dictionary (for CSV export etc.).
        """
        return {
            "mean_error":              self.mean_error,
            "mean_absolute_error":     self.mean_absolute_error,
            "mean_squared_error":      self.mean_squared_error,
            "root_mean_squared_error": self.root_mean_squared_error,
            "std_error":               self.std_error,
            "correlation":             self.correlation,
            "CI_95_upper":             self.CI95[0],
            "CI_95_lower":             self.CI95[1],
        }

    
    # Single-method plots
    

    def scatter_plot(
        self,
        x_label: str       = "Ground Truth",
        y_label: str       = "Estimate",
        figure_size: tuple = (5, 5),
        show_legend: bool  = True,
        the_title: str     = "",
        file_name: str     = "scatter_plot.pdf",
        is_journal: bool   = False,
    ) -> None:
        """
        Scatter plot of estimate vs. ground-truth values.
        """
        if is_journal:
            matplotlib.rcParams["pdf.fonttype"] = 42
            matplotlib.rcParams["ps.fonttype"]  = 42

        gold_j = self._jitter(self.gold_std.copy())
        new_j  = self._jitter(self.new_measure.copy())

        fig, ax = plt.subplots(figsize=figure_size)

        if len(gold_j) >= 10:
            z  = gaussian_kde(
                np.vstack([gold_j, new_j]))(
                np.vstack([gold_j, new_j]))
            sc = ax.scatter(gold_j, new_j, c=z,
                            s=40, cmap="plasma")
            plt.colorbar(sc, ax=ax,
                         label="Point density")
        else:
            ax.scatter(gold_j, new_j, s=40,
                       color="#5B9BD5",
                       edgecolors="black",
                       linewidths=0.5)

        lim_min = min(gold_j.min(), new_j.min())
        lim_max = max(gold_j.max(), new_j.max())
        margin  = (lim_max - lim_min) * 0.05
        eq_vals = np.array(
            [lim_min - margin, lim_max + margin])
        ax.plot(eq_vals, eq_vals, "--", color="black",
                linewidth=1, label="Line of equality")

        ax.set_xlim(eq_vals)
        ax.set_ylim(eq_vals)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(the_title)
        if show_legend:
            ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.4)

        save_file = os.path.join(
            self.save_path, file_name)
        plt.savefig(save_file, bbox_inches="tight",
                    dpi=300)
        plt.close(fig)
        print(f"Saved: {save_file}")

    def difference_plot(
        self,
        x_label: str       = "Mean of Ground Truth and Estimate",
        y_label: str       = "Difference (Ground Truth - Estimate)",
        figure_size: tuple = (5, 5),
        show_legend: bool  = True,
        the_title: str     = "",
        file_name: str     = "bland_altman_difference_plot.pdf",
        is_journal: bool   = False,
    ) -> None:
        """
        Classic Bland-Altman difference plot.
        """
        if is_journal:
            matplotlib.rcParams["pdf.fonttype"] = 42
            matplotlib.rcParams["ps.fonttype"]  = 42

        diffs = self.gold_std - self.new_measure
        avgs  = (self.gold_std + self.new_measure) / 2.0

        fig, ax = plt.subplots(figsize=figure_size)

        if len(avgs) >= 10:
            z = gaussian_kde(
                np.vstack([avgs, diffs]))(
                np.vstack([avgs, diffs]))
            ax.scatter(avgs, diffs, c=z, s=40,
                       cmap="plasma",
                       label="Observations")
        else:
            ax.scatter(avgs, diffs, s=40,
                       color="#5B9BD5",
                       edgecolors="black",
                       linewidths=0.5,
                       label="Observations")

        ax.axhline(self.mean_error, color="black",
                   linewidth=1.2,
                   label=f"Mean error = "
                         f"{self.mean_error:.2f}")
        ax.axhline(self.CI95[0], color="black",
                   linestyle="--", linewidth=0.9,
                   label=f"+95% LoA = "
                         f"{self.CI95[0]:.2f}")
        ax.axhline(self.CI95[1], color="black",
                   linestyle="--", linewidth=0.9,
                   label=f"-95% LoA = "
                         f"{self.CI95[1]:.2f}")

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(the_title)
        ax.grid(True, linewidth=0.4)
        if show_legend:
            ax.legend(fontsize=8)

        save_file = os.path.join(
            self.save_path, file_name)
        plt.savefig(save_file, bbox_inches="tight",
                    dpi=300)
        plt.close(fig)
        print(f"Saved: {save_file}")

    
    # Private helpers
    

    @staticmethod
    def _to_series(data, name: str) -> pd.Series:
        """
        Convert list, numpy array, or Series.
        """
        if isinstance(data, pd.Series):
            return data.reset_index(drop=True)
        if isinstance(data, (list, np.ndarray)):
            return pd.Series(data, name=name,
                             dtype=float)
        raise TypeError(
            f"{name} must be a list, numpy array, "
            f"or pandas Series, got {type(data)}."
        )

    @staticmethod
    def _jitter(arr: pd.Series) -> pd.Series:
        """
        Add tiny random noise for visibility.
        """
        data_range = arr.max() - arr.min()
        if data_range == 0:
            return arr
        return arr + np.random.randn(len(arr)) \
            * 0.01 * data_range



# Combined plots – all methods in one figure


def combined_scatter_plot(
    methods_data: dict,
    the_title: str = "",
    save_path: str = "results/summary",
    file_name: str = "scatter_combined.pdf",
    x_label: str   = "Ground Truth [BPM]",
    y_label: str   = "Estimated [BPM]",
    figure_size: tuple = (6, 6),
) -> None:
    """
    Scatter plot with all methods in one figure.
    
    methods_data : dict
        {method_name: (gt_array, est_array), ...}
    """
    os.makedirs(save_path, exist_ok=True)
    fig, ax = plt.subplots(figsize=figure_size)

    all_vals = []

    for method_name, (gt, est) in methods_data.items():
        gt  = np.asarray(gt, dtype=float)
        est = np.asarray(est, dtype=float)

        valid = ~(np.isnan(gt) | np.isnan(est))
        gt, est = gt[valid], est[valid]

        if len(gt) == 0:
            continue

        all_vals.extend(gt)
        all_vals.extend(est)

        color  = METHOD_COLORS.get(
            method_name, "#999999")
        marker = METHOD_MARKERS.get(
            method_name, "o")

        ax.scatter(gt, est, s=50, color=color,
                   marker=marker,
                   edgecolors="black",
                   linewidths=0.5, alpha=0.8,
                   label=method_name, zorder=3)

    # Line of equality
    if all_vals:
        lim_min = min(all_vals)
        lim_max = max(all_vals)
        margin  = (lim_max - lim_min) * 0.08
        eq_vals = np.array(
            [lim_min - margin, lim_max + margin])
        ax.plot(eq_vals, eq_vals, "--", color="black",
                linewidth=1, label="Line of equality",
                zorder=1)
        ax.set_xlim(eq_vals)
        ax.set_ylim(eq_vals)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(the_title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linewidth=0.4)

    path = os.path.join(save_path, file_name)
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def combined_difference_plot(
    methods_data: dict,
    the_title: str = "",
    save_path: str = "results/summary",
    file_name: str = "bland_altman_combined.pdf",
    x_label: str   = "Mean of GT and Estimate [BPM]",
    y_label: str   = "Difference (GT - Estimate) [BPM]",
    figure_size: tuple = (7, 5),
) -> None:
    """
    Bland-Altman difference plot with all methods
    in one figure. Each method gets its own colour,
    marker, and LoA lines.
    
    methods_data : dict
        {method_name: (gt_array, est_array), ...}
    """
    os.makedirs(save_path, exist_ok=True)
    fig, ax = plt.subplots(figsize=figure_size)

    for method_name, (gt, est) in methods_data.items():
        gt  = np.asarray(gt, dtype=float)
        est = np.asarray(est, dtype=float)

        valid = ~(np.isnan(gt) | np.isnan(est))
        gt, est = gt[valid], est[valid]

        if len(gt) == 0:
            continue

        diffs = gt - est
        avgs  = (gt + est) / 2.0
        mean_err = float(diffs.mean())
        std_err  = float(diffs.std())
        upper = mean_err + 1.96 * std_err
        lower = mean_err - 1.96 * std_err

        color  = METHOD_COLORS.get(
            method_name, "#999999")
        marker = METHOD_MARKERS.get(
            method_name, "o")

        # Data points
        ax.scatter(avgs, diffs, s=50, color=color,
                   marker=marker,
                   edgecolors="black",
                   linewidths=0.5, alpha=0.8,
                   label=f"{method_name} "
                         f"(bias={mean_err:+.1f})",
                   zorder=3)

        # Mean error line
        ax.axhline(mean_err, color=color,
                   linewidth=1.2, zorder=2)

        # LoA lines
        ax.axhline(upper, color=color,
                   linestyle="--", linewidth=0.8,
                   alpha=0.6, zorder=2)
        ax.axhline(lower, color=color,
                   linestyle="--", linewidth=0.8,
                   alpha=0.6, zorder=2)

    ax.axhline(0, color="black", linewidth=0.5,
               linestyle=":", zorder=1)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(the_title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linewidth=0.4)

    path = os.path.join(save_path, file_name)
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")



# Quick test


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    gt  = rng.normal(75, 10, 50)

    combined_difference_plot(
        methods_data={
            "thermal_mean": (gt, gt + rng.normal(2, 5, 50)),
            "ica":          (gt, gt + rng.normal(-3, 8, 50)),
            "garbey":       (gt, gt + rng.normal(1, 4, 50)),
        },
        the_title="Heart Rate – BP4D+",
        save_path="/Users/valeriamoltschanov/Desktop/bland_altman_test",
        file_name="hr_combined_difference.pdf",
    )
    combined_scatter_plot(
        methods_data={
            "thermal_mean": (gt, gt + rng.normal(2, 5, 50)),
            "ica":          (gt, gt + rng.normal(-3, 8, 50)),
            "garbey":       (gt, gt + rng.normal(1, 4, 50)),
        },
        the_title="Heart Rate – BP4D+",
        save_path="/Users/valeriamoltschanov/Desktop/bland_altman_test",
        file_name="hr_combined_scatter.pdf",
    )
