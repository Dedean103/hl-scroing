#!/usr/bin/env python3
"""
Utility script to run the VipHL strategy with the dynamic (m, n) detector and
visualize trade executions plus HL lines in a single pass.

This combines the old plotting helpers into one up-to-date entry point that
understands the latest Settings/Strategy parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional, Tuple

import backtrader as bt
from backtrader import num2date
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

VIPHL_REPO = REPO_ROOT / "viphl-source-code"
VIPHL_INDICATORS = VIPHL_REPO / "indicators"

for path in (VIPHL_REPO, VIPHL_INDICATORS, ROOT):
    if path.is_dir():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from viphl_strategy_scoring import VipHLStrategy, load_data_from_csv  # noqa: E402


DEFAULT_STRATEGY_CONFIG: Dict[str, Any] = {
    # HL window defaults (also used as fallbacks if dynamic expansion halts early)
    "high_by_point_n": 4,
    "high_by_point_m": 4,
    "low_by_point_n": 4,
    "low_by_point_m": 4,
    "high_by_point_n_on_trend": 4,
    "high_by_point_m_on_trend": 4,
    "low_by_point_n_on_trend": 4,
    "low_by_point_m_on_trend": 4,
    # Dynamic detector caps
    "mn_start_point_high": 10,
    "mn_start_point_low": 10,
    "mn_cap_high": 20,
    "mn_cap_low": 20,
    "mn_start_point_high_trend": 10,
    "mn_start_point_low_trend": 10,
    "mn_cap_high_trend": 20,
    "mn_cap_low_trend": 20,
    # Scoring
    "max_mn_cap": 20,
    "power_scaling_factor": 1.5,
    "high_score_scaling_factor": 0.5,
    "low_score_scaling_factor": 0.5,
    "on_trend_ratio": 1.0,
    # Misc
    "mintick": 0.01,
    "debug_mode": False,
    "debug_log_path": "",
}


def resolve_csv_path(csv_path: str) -> Path:
    """Return an existing CSV path, searching relative to this module if needed."""
    candidate = Path(csv_path)
    if candidate.is_file():
        return candidate

    fallback = ROOT / csv_path
    if fallback.is_file():
        return fallback

    raise FileNotFoundError(f"Could not find CSV at '{csv_path}' or '{fallback}'.")


def build_strategy_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI overrides into the default strategy configuration."""
    config = dict(DEFAULT_STRATEGY_CONFIG)
    config["mintick"] = args.mintick

    config["mn_start_point_high"] = args.mn_start_normal
    config["mn_start_point_low"] = args.mn_start_normal
    config["mn_cap_high"] = args.mn_cap_normal
    config["mn_cap_low"] = args.mn_cap_normal

    config["mn_start_point_high_trend"] = args.mn_start_trend
    config["mn_start_point_low_trend"] = args.mn_start_trend
    config["mn_cap_high_trend"] = args.mn_cap_trend
    config["mn_cap_low_trend"] = args.mn_cap_trend

    # Keep static fallback windows aligned with the starting dynamic window
    for key in (
        "high_by_point_n",
        "high_by_point_m",
        "low_by_point_n",
        "low_by_point_m",
        "high_by_point_n_on_trend",
        "high_by_point_m_on_trend",
        "low_by_point_n_on_trend",
        "low_by_point_m_on_trend",
    ):
        config[key] = args.static_window if args.static_window else config[key]

    config["power_scaling_factor"] = args.power_scaling_factor
    config["high_score_scaling_factor"] = args.high_score_scaling_factor
    config["low_score_scaling_factor"] = args.low_score_scaling_factor
    config["on_trend_ratio"] = args.on_trend_ratio
    config["debug_mode"] = bool(args.debug)
    config["debug_log_path"] = args.debug_log

    return config


def run_strategy_and_plot(
    csv_file: Path,
    save_plot: bool = True,
    show_plot: bool = True,
    output_dir: Optional[Path] = None,
    strategy_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[VipHLStrategy, bt.Cerebro]:
    """Execute VipHLStrategy on the provided CSV and plot/optionally save results."""
    dataframe = load_data_from_csv(str(csv_file))
    dataframe.index = pd.to_datetime(dataframe.index)

    cerebro = bt.Cerebro()
    pandas_data = bt.feeds.PandasData(dataname=dataframe)
    cerebro.adddata(pandas_data)
    cerebro.broker.set_coc(True)

    strategy_args = strategy_kwargs or DEFAULT_STRATEGY_CONFIG
    cerebro.addstrategy(VipHLStrategy, **strategy_args)

    print(f"Running VipHL strategy on {csv_file}...")
    results = cerebro.run()
    strat: VipHLStrategy = results[0]

    print("\n========== Strategy Metrics ==========")
    for key, value in strat.result.items():
        print(f"{key:<24}: {value}")
    print("======================================\n")

    ticker = csv_file.stem.upper()
    title = f"{ticker} VipHL Strategy — Dynamic (m, n) Results"
    save_filename = None

    if save_plot:
        output_dir = output_dir or csv_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        save_filename = output_dir / f"{ticker}_viphl_trades.png"

    strategy_params = {
        "mn_start_point_high": strat.params.mn_start_point_high,
        "mn_start_point_low": strat.params.mn_start_point_low,
        "mn_cap_high": strat.params.mn_cap_high,
        "mn_cap_low": strat.params.mn_cap_low,
        "mn_start_point_high_trend": strat.params.mn_start_point_high_trend,
        "mn_start_point_low_trend": strat.params.mn_start_point_low_trend,
        "mn_cap_high_trend": strat.params.mn_cap_high_trend,
        "mn_cap_low_trend": strat.params.mn_cap_low_trend,
        "power_scaling_factor": strat.params.power_scaling_factor,
        "high_score_scaling_factor": strat.params.high_score_scaling_factor,
        "low_score_scaling_factor": strat.params.low_score_scaling_factor,
        "on_trend_ratio": strat.params.on_trend_ratio,
        "enable_hl_byp_scoring": strat.params.enable_hl_byp_scoring,
        "debug_log_path": strat.params.debug_log_path,
    }

    plot_trade_results(
        dataframe=dataframe,
        trade_list=strat.trade_list,
        lines_info=strat.lines_info,
        result_stats=strat.result,
        title=title,
        save_filename=save_filename,
        strategy_params=strategy_params,
        show_plot=show_plot,
    )

    return strat, cerebro


def plot_trade_results(
    dataframe,
    trade_list,
    lines_info,
    result_stats,
    title: str = "VipHL Trades",
    save_filename: Optional[Path] = None,
    strategy_params: Optional[Dict[str, Any]] = None,
    show_plot: bool = True,
):
    """Render VipHL trades, HL lines, and strategy statistics."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(20, 10))

    x_axis = dataframe.index.to_numpy()
    close_values = dataframe["close"].to_numpy()

    ax.plot(
        x_axis,
        close_values,
        label="Close Price",
        color="darkblue",
        linewidth=1.5,
        alpha=0.6,
        zorder=1,
    )

    if lines_info:
        for hl_value, start_idx, end_idx in lines_info:
            start_idx = min(start_idx, len(dataframe) - 1)
            end_idx = min(end_idx, len(dataframe) - 1)
            ax.plot(
                [dataframe.index[start_idx], dataframe.index[end_idx]],
                [hl_value, hl_value],
                color="purple",
                linestyle="--",
                linewidth=1.5,
                alpha=0.4,
                zorder=2,
            )
        ax.plot([], [], color="purple", linestyle="--", linewidth=1.5, alpha=0.4, label="VipHL Lines")

    price_min = dataframe["close"].min()
    price_max = dataframe["close"].max()
    price_range = price_max - price_min

    base_entry_size = min((trade.total_entry_size for trade in trade_list if trade.total_entry_size > 0), default=1)
    if not base_entry_size:
        base_entry_size = 1

    for trade in trade_list:
        entry_date = num2date(trade.entry_time)
        entry_price = trade.entry_price
        scale = max(1.0, trade.total_entry_size / base_entry_size)
        pnl = trade.pnl
        marker_size = 50 + (scale - 1.0) * 50

        ax.scatter(
            entry_date,
            entry_price,
            s=marker_size,
            color="blue",
            marker="^",
            alpha=0.3,
            edgecolors="darkblue",
            linewidths=1.5,
            zorder=5,
        )

        if trade.is_open:
            continue

        exit_color = "green" if pnl > 0 else "red"
        exit_dark_color = "darkgreen" if pnl > 0 else "darkred"

        if trade.first_time > 0:
            first_exit_date = num2date(trade.first_time)
            first_exit_price = entry_price * (1 + trade.first_return / 100)
            ax.scatter(
                first_exit_date,
                first_exit_price,
                s=marker_size,
                color=exit_color,
                marker="v",
                alpha=0.3,
                edgecolors=exit_dark_color,
                linewidths=1.5,
                zorder=5,
            )
            ax.plot(
                [entry_date, first_exit_date],
                [entry_price, first_exit_price],
                color=exit_color,
                linestyle=":",
                linewidth=1.5,
                alpha=0.5,
                zorder=3,
            )

        if trade.take_profit and trade.second_time > 0:
            second_exit_date = num2date(trade.second_time)
            second_exit_price = entry_price * (1 + trade.second_return / 100)
            ax.scatter(
                second_exit_date,
                second_exit_price,
                s=marker_size * 0.8,
                color=exit_color,
                marker="v",
                alpha=0.25,
                edgecolors=exit_dark_color,
                linewidths=1.5,
                zorder=5,
            )
            if trade.first_time > 0:
                first_exit_date = num2date(trade.first_time)
                first_exit_price = entry_price * (1 + trade.first_return / 100)
                ax.plot(
                    [first_exit_date, second_exit_date],
                    [first_exit_price, second_exit_price],
                    color=exit_color,
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.4,
                    zorder=3,
                )

    stats_text = (
        "Trade Statistics:\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"Total PnL: {result_stats.get('Total Pnl%', 0):.2f}%\n"
        f"Avg PnL / Trade: {result_stats.get('Avg Pnl% per entry', 0):.2f}%\n"
        f"Trade Count: {result_stats.get('Trade Count', 0)}\n"
        f"Win Rate: {result_stats.get('Winning entry%', 0):.2f}%\n"
        f"Avg Winner: {result_stats.get('Avg Winner%', 0):.2f}%\n"
        f"Avg Loser: {result_stats.get('Avg Loser%', 0):.2f}%\n"
        f"Fit Score: {result_stats.get('Fit Score', 0):.2f}\n"
        f"PnL Scale: {result_stats.get('Scale', 0):.2f}\n"
    )

    if strategy_params:
        params_text = (
            "\n━━━━━━━━━━━━━━━━━━━━\n"
            "Strategy Parameters:\n"
            f"Normal mn: start {strategy_params['mn_start_point_high']}/{strategy_params['mn_start_point_low']} "
            f"cap {strategy_params['mn_cap_high']}/{strategy_params['mn_cap_low']}\n"
            f"Trend mn: start {strategy_params['mn_start_point_high_trend']}/{strategy_params['mn_start_point_low_trend']} "
            f"cap {strategy_params['mn_cap_high_trend']}/{strategy_params['mn_cap_low_trend']}\n"
            f"Power Scaling k: {strategy_params['power_scaling_factor']:.2f}\n"
            f"High/Low Score Weights: {strategy_params['high_score_scaling_factor']:.2f} / "
            f"{strategy_params['low_score_scaling_factor']:.2f}\n"
            f"On-Trend Ratio: {strategy_params['on_trend_ratio']:.2f}\n"
            f"HL byP Scoring: {'Enabled' if strategy_params['enable_hl_byp_scoring'] else 'Disabled'}"
        )
        stats_text += params_text

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="left",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8, edgecolor="black", linewidth=1.5),
    )

    legend_elements = [
        Line2D([0], [0], marker="^", color="w", label="Entry", markerfacecolor="blue", markeredgecolor="darkblue", markersize=10, markeredgewidth=1.5),
        Line2D([0], [0], marker="v", color="w", label="Exit (Profit)", markerfacecolor="green", markeredgecolor="darkgreen", markersize=10, markeredgewidth=1.5),
        Line2D([0], [0], marker="v", color="w", label="Exit (Loss)", markerfacecolor="red", markeredgecolor="darkred", markersize=10, markeredgewidth=1.5),
        Line2D([0], [0], color="darkblue", linewidth=1.5, alpha=0.6, label="Close Price"),
    ]

    if lines_info:
        legend_elements.append(
            Line2D([0], [0], color="purple", linestyle="--", linewidth=1.5, alpha=0.4, label="VipHL Lines"),
        )

    ax.legend(handles=legend_elements, loc="upper right", fontsize=11)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
    ax.set_xlabel("Date", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(price_min - price_range * 0.05, price_max + price_range * 0.1)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    if save_filename:
        save_path = Path(save_filename)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Trade visualization saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VipHLStrategy with dynamic (m, n) sizing and plot the resulting trades.",
    )
    parser.add_argument("--csv", default="BTC.csv", help="Path to the OHLCV CSV (default: %(default)s).")
    parser.add_argument("--mintick", type=float, default=0.01, help="Minimum tick size passed to the strategy.")
    parser.add_argument("--mn-start-normal", type=int, default=4, help="Starting m/n window for normal pivots.")
    parser.add_argument("--mn-cap-normal", type=int, default=20, help="Maximum m/n window for normal pivots.")
    parser.add_argument("--mn-start-trend", type=int, default=4, help="Starting m/n window for trending pivots.")
    parser.add_argument("--mn-cap-trend", type=int, default=20, help="Maximum m/n window for trending pivots.")
    parser.add_argument("--static-window", type=int, default=0, help="Optional override for static fallback windows.")
    parser.add_argument("--power-scaling-factor", type=float, default=1.5, help="k exponent for HL scoring.")
    parser.add_argument("--high-score-scaling-factor", type=float, default=0.5, help="Weight on high pivots.")
    parser.add_argument("--low-score-scaling-factor", type=float, default=0.5, help="Weight on low pivots.")
    parser.add_argument("--on-trend-ratio", type=float, default=1.0, help="Weight boost applied in trending mode.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output for ByPoint/HL construction.")
    parser.add_argument("--debug-log", default="", help="Path to append debug markdown output.")
    parser.add_argument("--no-save", action="store_true", help="Skip saving the PNG output.")
    parser.add_argument("--no-show", action="store_true", help="Skip displaying the matplotlib window.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for plot output; defaults to the CSV directory.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    csv_path = resolve_csv_path(args.csv)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    strategy_kwargs = build_strategy_kwargs(args)

    run_strategy_and_plot(
        csv_file=csv_path,
        save_plot=not args.no_save,
        show_plot=not args.no_show,
        output_dir=output_dir,
        strategy_kwargs=strategy_kwargs,
    )


if __name__ == "__main__":
    main()
