"""
results_table.py

Collects evaluation results and produces a formatted
comparison table across all methods and datasets.

Saved as CSV (for processing) and PDF (for report).

Output CSV columns
------------------
    Dataset | Method | Target | MAE | MAE_SE | RMSE |
    MAPE | Pearson | n
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Columns shown in the final table (in this order)
_DISPLAY_COLS = [
    "Dataset", "Method", "Target",
    "MAE", "MAE_SE", "RMSE", "MAPE", "Pearson", "n",
]

# Column labels for PDF/print output
_COL_LABELS = {
    "Dataset": "Dataset",
    "Method":  "Method",
    "Target":  "Target",
    "MAE":     "MAE [BPM]",
    "MAE_SE":  "± SE",
    "RMSE":    "RMSE",
    "MAPE":    "MAPE [%]",
    "Pearson": "Pearson r",
    "n":       "n",
}


class ResultsTable:
    """
    Accumulates per-method evaluation results and
    renders a comparison table.
    """

    def __init__(self, save_path: str = "results"):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self._rows: list[dict] = []

    def add(self, dataset: str, method: str,
            target: str, metrics: dict) -> None:
        """
        Add one result row to the table.

        Parameters
        ----------
        dataset : str  -- e.g. "BP4D+"
        method  : str  -- e.g. "ICA", "Garbey"
        target  : str  -- "HR" or "RR"
        metrics : dict -- output of evaluate_hr/rr()
        """
        row = {
            "Dataset": dataset,
            "Method":  method,
            "Target":  target,
            "MAE":     round(metrics.get(
                "MAE", float("nan")), 3),
            "MAE_SE":  round(metrics.get(
                "MAE_SE", float("nan")), 3),
            "RMSE":    round(metrics.get(
                "RMSE", float("nan")), 3),
            "MAPE":    round(metrics.get(
                "MAPE", float("nan")), 2),
            "Pearson": round(metrics.get(
                "Pearson", float("nan")), 3),
            "n":       int(metrics.get("n", 0)),
        }
        self._rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return results as pandas DataFrame.
        """
        if not self._rows:
            return pd.DataFrame(columns=_DISPLAY_COLS)
        df = pd.DataFrame(self._rows)
        for col in _DISPLAY_COLS:
            if col not in df.columns:
                df[col] = float("nan")
        return df[_DISPLAY_COLS].sort_values(
            ["Target", "Dataset", "Method"])

    def print(self) -> None:
        """
        Print results table to stdout.
        """
        df = self.to_dataframe()
        if df.empty:
            print("No results to display.")
            return
        df_display = df.rename(columns=_COL_LABELS)
        print("\n" + "=" * 80)
        print("  Evaluation Results")
        print("=" * 80)
        print(df_display.to_string(index=False))
        print("=" * 80 + "\n")

    def save_csv(self,
                 filename: str = "results.csv") -> str:
        """
        Save results as CSV file.
        """
        df = self.to_dataframe()
        path = os.path.join(self.save_path, filename)
        df.to_csv(path, index=False)
        print(f"CSV saved: {path}")
        return path

    def save_pdf(self,
                 filename: str = "results_table.pdf"
                 ) -> str:
        """
        Save results as a formatted PDF table.
        - HR and RR sections visually separated
        - Best MAE per group highlighted in green
        - Standard errors shown next to MAE
        """
        df = self.to_dataframe()
        if df.empty:
            print("No results to save.")
            return ""

        # Merge MAE and MAE_SE into one column
        df_pdf = df.copy()
        df_pdf["MAE ± SE"] = df_pdf.apply(
            lambda r: f"{r['MAE']:.3f} ± "
                      f"{r['MAE_SE']:.3f}",
            axis=1,
        )

        pdf_cols = [
            "Dataset", "Method", "Target",
            "MAE ± SE", "RMSE", "MAPE", "Pearson", "n",
        ]
        col_labels = [
            "Dataset", "Method", "Target",
            "MAE ± SE [BPM]", "RMSE",
            "MAPE [%]", "Pearson r", "n",
        ]

        df_pdf = df_pdf[pdf_cols]
        n_rows = len(df_pdf)

        fig_h = max(2.5, 0.5 * n_rows + 2.0)
        fig, ax = plt.subplots(figsize=(12, fig_h))
        ax.axis("off")

        tbl = ax.table(
            cellText=df_pdf.values,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.auto_set_column_width(
            col=list(range(len(pdf_cols))))

        # Style header
        for j in range(len(pdf_cols)):
            cell = tbl[0, j]
            cell.set_facecolor("#2C2C2A")
            cell.set_text_props(
                color="white", fontweight="bold")
            cell.set_edgecolor("#444441")

        # Highlight best MAE per group
        _best_mae = {}
        for _, grp in df.groupby(["Target", "Dataset"]):
            idx = grp["MAE"].idxmin()
            _best_mae[idx] = True

        for i, (orig_idx, row) in enumerate(
                df.iterrows()):
            row_num = i + 1
            bg = "#F9F8F5" if i % 2 == 0 else "white"
            for j in range(len(pdf_cols)):
                cell = tbl[row_num, j]
                cell.set_edgecolor("#D3D1C7")
                if orig_idx in _best_mae:
                    cell.set_facecolor("#EAF3DE")
                else:
                    cell.set_facecolor(bg)

        # Title
        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M")
        ax.set_title(
            f"Thermal Vital Signs Toolbox — "
            f"Evaluation Results\n"
            f"Generated: {timestamp}",
            fontsize=10, pad=16, loc="left",
        )

        plt.tight_layout()
        path = os.path.join(self.save_path, filename)
        plt.savefig(path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"PDF saved: {path}")
        return path
