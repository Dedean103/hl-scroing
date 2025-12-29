#!/usr/bin/env python3
"""Plot Day-End PnL% and Remaining Fund from a VipHL daily summary CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_daily_summary(
    csv_path: Path,
    output_path: Optional[Path] = None,
    show_plot: bool = False,
) -> Path:
    """Render Day End PnL% and Remaining Fund over time using dual y-axes."""
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Daily summary CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["Date"])
    if df.empty:
        raise ValueError(f"{csv_path} is empty; nothing to plot.")

    fig, ax_left = plt.subplots(figsize=(14, 6))

    ax_left.plot(df["Date"], df["Day End PnL%"], color="tab:blue", linewidth=2.0, label="Day End PnL%")
    ax_left.set_xlabel("Date")
    ax_left.set_ylabel("Day End PnL (%)", color="tab:blue")
    ax_left.tick_params(axis="y", labelcolor="tab:blue")
    ax_left.grid(True, linestyle="--", alpha=0.3)

    #ax_right = ax_left.twinx()
    #ax_right.plot(df["Date"], df["Remaining Fund"], color="tab:green", linewidth=2.0, label="Remaining Fund")
    #ax_right.set_ylabel("Remaining Fund (USD)", color="tab:green")
    #ax_right.tick_params(axis="y", labelcolor="tab:green")

    title = f"{csv_path.stem.replace('_viphl_daily_summary_', ' ')} — Daily Equity"
    ax_left.set_title(title.strip())
    fig.autofmt_xdate()

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    ax_left.legend(lines_left, labels_left, loc="upper left")

    if output_path is None:
        output_path = csv_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return output_path

token = '50_20_1_bar900_20251211142626'

def main() -> None:
    csv_path = Path(
        r"C:\Users\zt010\OneDrive - Unitex International Button Accessories Ltd\pn\scoring\hl-scroing\results\50_20_1_bar900_20251211142626\BTC_viphl_daily_summary_50_20_1_bar900_20251211142626.csv"
    )
    plot_daily_summary(csv_path)


if __name__ == "__main__":
    main()
