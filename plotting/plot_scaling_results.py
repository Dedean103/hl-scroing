#!/usr/bin/env python3
"""Batch-generate scaling comparison plots from VipHL daily summary CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_daily_increment import plot_daily_increment


def _find_summary_csv(directory: Path) -> Path:
    """Locate a single *_viphl_daily_summary_*.csv file within ``directory``."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    matches = sorted(directory.glob("*_viphl_daily_summary_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No daily summary CSV found under {directory}")
    if len(matches) > 1:
        print(f"Warning: multiple daily summaries in {directory}; using {matches[-1].name}")
    return matches[-1]


def _load_summary(csv_path: Path) -> pd.DataFrame:
    """Return dataframe with Date, Day End PnL%, and Daily Change%."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    if df.empty:
        raise ValueError(f"{csv_path} is empty.")
    if "Day End PnL%" not in df:
        raise KeyError(f"'Day End PnL%' column missing from {csv_path}")

    summary = df[["Date", "Day End PnL%"]].copy()
    if "Daily Change%" in df:
        summary["Daily Change%"] = df["Daily Change%"].astype(float)
    elif "Remaining Fund" in df:
        fund = df["Remaining Fund"].astype(float)
        prev = fund.shift(1).replace(0, pd.NA)
        summary["Daily Change%"] = fund.subtract(prev).div(prev).mul(100).fillna(0.0)
    else:
        raise KeyError(f"Need either 'Daily Change%' or 'Remaining Fund' columns in {csv_path}")

    return summary


def _plot_comparison(
    best_df: pd.DataFrame,
    not_df: pd.DataFrame,
    best_label: str,
    not_label: str,
    output_path: Path,
) -> Path:
    """Create a comparison plot for Day End PnL% and Daily Change%."""
    merged = (
        best_df.set_index("Date")
        .join(not_df.set_index("Date"), how="outer", lsuffix="_best", rsuffix="_not")
        .sort_index()
        .reset_index()
    )

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax_top.plot(merged["Date"], merged["Day End PnL%_best"], label=best_label, color="tab:blue", linewidth=2.0)
    ax_top.plot(merged["Date"], merged["Day End PnL%_not"], label=not_label, color="tab:orange", linewidth=2.0)
    ax_top.set_ylabel("Day End PnL (%)")
    ax_top.set_title(f"{best_label} vs {not_label} — Day End PnL%")
    ax_top.grid(True, linestyle="--", alpha=0.3)
    ax_top.legend(loc="upper left")

    ax_bottom.plot(merged["Date"], merged["Daily Change%_best"], label=f"{best_label} Daily%", color="tab:green")
    ax_bottom.plot(merged["Date"], merged["Daily Change%_not"], label=f"{not_label} Daily%", color="tab:red")
    ax_bottom.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_bottom.set_xlabel("Date")
    ax_bottom.set_ylabel("Daily Change (%)")
    ax_bottom.set_title("Daily Increment (%)")
    ax_bottom.grid(True, linestyle="--", alpha=0.3)
    ax_bottom.legend(loc="upper left")

    fig.autofmt_xdate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved scaling comparison plot to {output_path}")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare best-scaling vs not-scaling VipHL runs.")
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "1217",
        help="Parent directory containing the scaling subdirectories.",
    )
    parser.add_argument(
        "--best-dir",
        type=str,
        default="best scaling",
        help="Subdirectory name for the best scaling run.",
    )
    parser.add_argument(
        "--not-dir",
        type=str,
        default="not scaling",
        help="Subdirectory name for the non-scaling run.",
    )
    parser.add_argument(
        "--price-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "BTC.csv",
        help="Path to BTC price history CSV for overlay plots.",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=None,
        help="Optional path for the combined comparison PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parent = Path(args.parent).expanduser().resolve()
    best_dir = parent / args.best_dir
    not_dir = parent / args.not_dir

    best_csv = _find_summary_csv(best_dir)
    not_csv = _find_summary_csv(not_dir)

    best_df = _load_summary(best_csv)
    not_df = _load_summary(not_csv)

    best_remaining_png, best_day_end_png, best_daily_png = plot_daily_increment(
        best_csv,
        price_csv=args.price_csv,
    )
    not_remaining_png, not_day_end_png, not_daily_png = plot_daily_increment(
        not_csv,
        price_csv=args.price_csv,
    )

    comparison_output = args.comparison_output
    if comparison_output is None:
        comparison_output = parent / "best_vs_not_scaling_summary.png"
    _plot_comparison(
        best_df=best_df,
        not_df=not_df,
        best_label=best_dir.name,
        not_label=not_dir.name,
        output_path=comparison_output,
    )

    print("Generated PNG files:")
    print(f"  {best_remaining_png}")
    print(f"  {best_day_end_png}")
    print(f"  {best_daily_png}")
    print(f"  {not_remaining_png}")
    print(f"  {not_day_end_png}")
    print(f"  {not_daily_png}")
    print(f"  {comparison_output}")


if __name__ == "__main__":
    main()
