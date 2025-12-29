#!/usr/bin/env python3
"""Generate remaining-fund, day-end PnL, and daily-increment plots with BTC overlays."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_PRICE_CSV = ROOT / "BTC.csv"


def _load_summary(csv_path: Path) -> pd.DataFrame:
    """Read VipHL daily summary, ensuring required columns are present."""
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Daily summary CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["Date"])
    if df.empty:
        raise ValueError(f"{csv_path} is empty; nothing to plot.")

    if "Remaining Fund" not in df.columns:
        raise KeyError("CSV missing 'Remaining Fund' column, required for plotting.")
    df["Remaining Fund"] = df["Remaining Fund"].astype(float)

    if "Day End PnL%" not in df.columns:
        raise KeyError("CSV missing 'Day End PnL%' column, required for plotting.")
    df["Day End PnL%"] = df["Day End PnL%"].astype(float)

    if "Daily Change%" not in df.columns:
        fund = df["Remaining Fund"]
        prev = fund.shift(1).replace(0, pd.NA)
        df["Daily Change%"] = fund.subtract(prev).div(prev).mul(100).fillna(0.0)
    else:
        df["Daily Change%"] = df["Daily Change%"].astype(float)

    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _load_btc_price(price_csv: Path) -> pd.Series:
    price_csv = Path(price_csv).expanduser().resolve()
    if not price_csv.is_file():
        raise FileNotFoundError(f"BTC price CSV not found: {price_csv}")

    price_df = pd.read_csv(price_csv, parse_dates=["datetime"])
    if "close" not in price_df.columns:
        raise KeyError("Price CSV must include a 'close' column.")

    price_df["Date"] = price_df["datetime"].dt.normalize()
    price_df.sort_values("Date", inplace=True)
    daily_price = price_df.groupby("Date")["close"].last()
    daily_price.name = "BTC Close"
    return daily_price


def _align_price(summary_dates: pd.Series, price_series: pd.Series) -> pd.Series:
    date_index = pd.DatetimeIndex(summary_dates)
    aligned = price_series.reindex(date_index)
    aligned = aligned.ffill().bfill()
    return pd.Series(aligned.to_numpy(), index=date_index, name="BTC Close")


def _plot_remaining_fund(summary_df: pd.DataFrame, price_series: pd.Series, output_path: Path, show_plot: bool) -> Path:
    fig, ax_left = plt.subplots(figsize=(14, 6))
    ax_left.plot(summary_df["Date"], summary_df["Remaining Fund"], color="tab:blue", linewidth=2.0, label="Remaining Fund")
    ax_left.set_xlabel("Date")
    ax_left.set_ylabel("Remaining Fund (USD)", color="tab:blue")
    ax_left.tick_params(axis="y", labelcolor="tab:blue")
    ax_left.grid(True, linestyle="--", alpha=0.3)

    ax_right = ax_left.twinx()
    ax_right.plot(summary_df["Date"], price_series, color="tab:orange", linewidth=1.8, label="BTC Close")
    ax_right.set_ylabel("BTC Close (USD)", color="tab:orange")
    ax_right.tick_params(axis="y", labelcolor="tab:orange")

    ax_left.set_title("Remaining Fund vs BTC Price")
    fig.autofmt_xdate()

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved remaining-fund plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def _plot_daily_increment(summary_df: pd.DataFrame, price_series: pd.Series, output_path: Path, show_plot: bool) -> Path:
    daily_change = summary_df["Daily Change%"]
    colors = ["tab:green" if value >= 0 else "tab:red" for value in daily_change]

    fig, ax_left = plt.subplots(figsize=(14, 6))
    ax_left.bar(summary_df["Date"], daily_change, color=colors, label="Daily Change%", alpha=0.8)
    ax_left.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax_left.set_xlabel("Date")
    ax_left.set_ylabel("Daily Change (%)", color="tab:gray")
    ax_left.grid(True, linestyle="--", alpha=0.3)

    ax_right = ax_left.twinx()
    ax_right.plot(summary_df["Date"], price_series, color="tab:purple", linewidth=1.5, label="BTC Close")
    ax_right.set_ylabel("BTC Close (USD)", color="tab:purple")
    ax_right.tick_params(axis="y", labelcolor="tab:purple")

    run_label = summary_df.attrs.get("run_label", "")
    title = f"{run_label} Daily Increment vs BTC Price".strip()
    ax_left.set_title(title or "Daily Increment vs BTC Price")
    fig.autofmt_xdate()

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved daily-increment plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def _plot_day_end_pnl(summary_df: pd.DataFrame, price_series: pd.Series, output_path: Path, show_plot: bool) -> Path:
    pnl = summary_df["Day End PnL%"]

    fig, ax_left = plt.subplots(figsize=(14, 6))
    ax_left.plot(summary_df["Date"], pnl, color="tab:blue", linewidth=2.0, label="Day End PnL%")
    ax_left.set_xlabel("Date")
    ax_left.set_ylabel("Day End PnL (%)", color="tab:blue")
    ax_left.tick_params(axis="y", labelcolor="tab:blue")
    ax_left.grid(True, linestyle="--", alpha=0.3)

    ax_right = ax_left.twinx()
    ax_right.plot(summary_df["Date"], price_series, color="tab:orange", linewidth=1.6, label="BTC Close")
    ax_right.set_ylabel("BTC Close (USD)", color="tab:orange")
    ax_right.tick_params(axis="y", labelcolor="tab:orange")

    run_label = summary_df.attrs.get("run_label", "")
    title = f"{run_label} Day End PnL% vs BTC Price".strip()
    ax_left.set_title(title or "Day End PnL% vs BTC Price")
    fig.autofmt_xdate()

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved day-end PnL plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def plot_daily_increment(
    csv_path: Path,
    price_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    show_plot: bool = False,
) -> Tuple[Path, Path, Path]:
    """Create remaining-fund, day-end PnL, and daily-increment PNGs with BTC price overlays."""
    summary_df = _load_summary(csv_path)
    price_series = _load_btc_price(price_csv or DEFAULT_PRICE_CSV)
    aligned_price = _align_price(summary_df["Date"], price_series)

    csv_path = Path(csv_path).expanduser().resolve()
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else csv_path.parent
    run_token = csv_path.stem.replace("BTC_viphl_daily_summary_", "")
    summary_df.attrs["run_label"] = run_token

    remaining_name = f"BTC_viphl_remaining_fund_with_price_{run_token}.png"
    day_end_name = f"BTC_viphl_day_end_pnl_with_price_{run_token}.png"
    daily_name = f"BTC_viphl_daily_increment_with_price_{run_token}.png"

    remaining_path = target_dir / remaining_name
    day_end_path = target_dir / day_end_name
    daily_path = target_dir / daily_name

    _plot_remaining_fund(summary_df, aligned_price, remaining_path, show_plot)
    _plot_day_end_pnl(summary_df, aligned_price, day_end_path, show_plot)
    _plot_daily_increment(summary_df, aligned_price, daily_path, show_plot)
    return remaining_path, day_end_path, daily_path


def _discover_summary_csv(directory: Path) -> Path:
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    matches = sorted(directory.glob("*_viphl_daily_summary_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No daily summary CSV found under {directory}")
    if len(matches) > 1:
        print(f"Warning: multiple daily summaries in {directory}; using {matches[-1].name}")
    return matches[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create remaining-fund and daily-increment plots with BTC overlays.")
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        help="Optional explicit BTC_viphl_daily_summary_*.csv path (repeatable).",
    )
    parser.add_argument(
        "--parent",
        type=Path,
        default=ROOT / "results" / "1217",
        help="Parent directory containing the default scaling subdirectories.",
    )
    parser.add_argument(
        "--best-dir",
        type=str,
        default="best scaling",
        help="Subdirectory for the best scaling run (used when --csv is omitted).",
    )
    parser.add_argument(
        "--not-dir",
        type=str,
        default="not scaling",
        help="Subdirectory for the non-scaling run (used when --csv is omitted).",
    )
    parser.add_argument(
        "--price-csv",
        type=Path,
        default=DEFAULT_PRICE_CSV,
        help="Path to BTC price history CSV (defaults to hl-scroing/BTC.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated PNGs (defaults to the CSV's directory).",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display matplotlib windows after saving the PNGs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths: List[Path] = []
    if args.csv:
        csv_paths.extend(Path(p).expanduser().resolve() for p in args.csv)
    else:
        parent = Path(args.parent).expanduser().resolve()
        csv_paths.append(_discover_summary_csv(parent / args.best_dir))
        csv_paths.append(_discover_summary_csv(parent / args.not_dir))

    for csv_path in csv_paths:
        remaining_path, day_end_path, daily_path = plot_daily_increment(
            csv_path=csv_path,
            price_csv=args.price_csv,
            output_dir=args.output_dir,
            show_plot=args.show_plot,
        )
        print("Generated plots:")
        print(f"  {remaining_path}")
        print(f"  {day_end_path}")
        print(f"  {daily_path}")


if __name__ == "__main__":
    main()
