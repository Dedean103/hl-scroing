#!/usr/bin/env python3
"""
Utility script to run the VipHL strategy with the dynamic (m, n) detector and
visualize trade executions plus HL lines in a single pass.

This combines the old plotting helpers into one up-to-date entry point that
understands the latest Settings/Strategy parameters.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from datetime import datetime
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

RESULTS_ROOT = ROOT / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

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
    "bar_count_to_by_point": 1000,
    # Scoring
    "max_mn_cap": 20,
    "power_scaling_factor": 1.5,
    "high_score_scaling_factor": 0.5,
    "low_score_scaling_factor": 0.5,
    "on_trend_ratio": 1.0,
    "enable_hl_byp_scoring": False,
    # Misc
    "mintick": 0.01,
    "debug_mode": True,
    "debug_log_path": str(RESULTS_ROOT),
    "lookback": 1000,
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
    config["lookback"] = args.lookback

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
    if args.debug_log:
        config["debug_log_path"] = args.debug_log
    if getattr(args, "enable_scoring", None) is not None:
        config["enable_hl_byp_scoring"] = bool(args.enable_scoring)
    config["bar_count_to_by_point"] = args.bar_count_to_by_point
    if args.start_date:
        config["plot_start_date"] = args.start_date
    if args.end_date:
        config["plot_end_date"] = args.end_date

    return config


def run_strategy_and_plot(
    csv_file: Path,
    save_plot: bool = True,
    show_plot: bool = False,
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

    strategy_args = dict(strategy_kwargs or DEFAULT_STRATEGY_CONFIG)
    plot_start = strategy_args.pop("plot_start_date", None)
    plot_end = strategy_args.pop("plot_end_date", None)

    run_token = "{normal}_{trend}_{scoreflag}_bar{bar_count}_{stamp}".format(
        normal=strategy_args.get("mn_start_point_high", "na"),
        trend=strategy_args.get("mn_start_point_high_trend", "na"),
        scoreflag=1 if strategy_args.get("enable_hl_byp_scoring") else 0,
        bar_count=strategy_args.get("bar_count_to_by_point", "na"),
        stamp=datetime.utcnow().strftime("%Y%m%d%H%M%S"),
    )

    base_output_root = output_dir if output_dir else RESULTS_ROOT
    output_root = base_output_root / run_token
    output_root.mkdir(parents=True, exist_ok=True)

    debug_log_path = strategy_args.get("debug_log_path")
    try:
        debug_log_root = Path(debug_log_path).expanduser().resolve() if debug_log_path else None
    except OSError:
        debug_log_root = None
    if debug_log_root is None or debug_log_root == RESULTS_ROOT.resolve():
        strategy_args["debug_log_path"] = str(output_root)
        debug_log_path = strategy_args["debug_log_path"]
    if debug_log_path:
        base_path = Path(debug_log_path).expanduser().resolve()
        if base_path.suffix:
            directory = base_path.parent
        else:
            directory = base_path
        directory.mkdir(parents=True, exist_ok=True)
        resolved_log_path = directory / f"debug_trace_{run_token}.md"
        header = f"# VipHL Debug Trace — {datetime.utcnow().date().isoformat()}\n\n"
        resolved_log_path.write_text(header, encoding="utf-8")
        strategy_args["debug_log_path"] = str(resolved_log_path)

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
        save_filename = output_root / f"{ticker}_viphl_trades_{run_token}.png"

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

    price_fig, _ = plot_trade_results(
        dataframe=dataframe,
        trade_list=strat.trade_list,
        lines_info=strat.lines_info,
        result_stats=strat.result,
        title=title,
        save_filename=save_filename,
        strategy_params=strategy_params,
        show_plot=show_plot,
        plot_start_date=plot_start,
        plot_end_date=plot_end,
    )

    pnl_history = getattr(strat, "pnl_history", [])
    pnl_plot_filename = None
    if save_plot:
        output_root.mkdir(parents=True, exist_ok=True)
        pnl_plot_filename = output_root / f"{ticker}_viphl_pnl_{run_token}.png"
    if pnl_history:
        plot_cumulative_pnl(
            pnl_history=pnl_history,
            title=f"{ticker} Cumulative PnL — Dynamic (m, n) Results",
            save_filename=pnl_plot_filename,
            show_plot=show_plot,
        )
    else:
        print("No PnL history available to plot cumulative curve.")

    export_trade_log(
        trade_list=strat.trade_list,
        output_dir=output_root,
        ticker=ticker,
        run_token=run_token,
    )
    export_daily_equity_summary(
        strategy=strat,
        price_index=dataframe.index,
        output_dir=output_root,
        ticker=ticker,
        run_token=run_token,
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
    plot_start_date: Optional[str] = None,
    plot_end_date: Optional[str] = None,
):
    """Render VipHL trades, HL lines, and strategy statistics."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(20, 10))

    filtered_df = dataframe.copy()
    if plot_start_date:
        start_dt = pd.to_datetime(plot_start_date)
        filtered_df = filtered_df[filtered_df.index >= start_dt]
    else:
        start_dt = filtered_df.index.min()
    if plot_end_date:
        end_dt = pd.to_datetime(plot_end_date)
        filtered_df = filtered_df[filtered_df.index <= end_dt]
    else:
        end_dt = filtered_df.index.max()
    if filtered_df.empty:
        filtered_df = dataframe
        start_dt = filtered_df.index.min()
        end_dt = filtered_df.index.max()

    x_axis = filtered_df.index.to_numpy()
    close_values = filtered_df["close"].to_numpy()

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

    price_min = filtered_df["close"].min()
    price_max = filtered_df["close"].max()
    price_range = price_max - price_min

    base_entry_size = min((trade.total_entry_size for trade in trade_list if trade.total_entry_size > 0), default=1)
    if not base_entry_size:
        base_entry_size = 1

    for trade in trade_list:
        entry_date = num2date(trade.entry_time)
        if entry_date < start_dt or entry_date > end_dt:
            continue
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
            if first_exit_date < start_dt or first_exit_date > end_dt:
                continue
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
            if second_exit_date < start_dt or second_exit_date > end_dt:
                continue
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
    ax.set_xlim(start_dt, end_dt)
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


def plot_cumulative_pnl(
    pnl_history,
    title: str,
    save_filename: Optional[Path] = None,
    show_plot: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Plot cumulative PnL percentage over time."""
    if not pnl_history:
        return None

    history = sorted(pnl_history, key=lambda point: point[0])
    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date) if end_date else None
    if start_dt is not None:
        history = [point for point in history if point[0] >= start_dt]
    if end_dt is not None:
        history = [point for point in history if point[0] <= end_dt]
    if not history:
        history = sorted(pnl_history, key=lambda point: point[0])
        start_dt = start_dt or history[0][0]
        end_dt = end_dt or history[-1][0]
    else:
        if start_dt is None:
            start_dt = history[0][0]
        if end_dt is None:
            end_dt = history[-1][0]

    dates = [point[0] for point in history]
    pnl_values = [point[1] for point in history]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dates, pnl_values, color="teal", linewidth=2.5, label="Cumulative PnL (%)")
    ax.fill_between(dates, pnl_values, color="teal", alpha=0.1)
    ax.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.6)

    if pnl_values:
        ax.annotate(
            f"{pnl_values[-1]:.2f}%",
            xy=(dates[-1], pnl_values[-1]),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            color="teal",
        )

    ax.set_xlim(start_dt, end_dt)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Cumulative PnL (%)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()

    if save_filename:
        save_path = Path(save_filename)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Cumulative PnL plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def export_trade_log(
    trade_list,
    output_dir: Optional[Path],
    ticker: str,
    run_token: str,
):
    """Write a CSV summarizing trade entry/exit timestamps and PnL."""
    if not trade_list:
        return None

    rows = []
    for idx, trade in enumerate(trade_list, start=1):
        entry_dt = num2date(trade.entry_time)
        first_exit_dt = num2date(trade.first_time) if trade.first_time else None
        second_exit_dt = num2date(trade.second_time) if trade.second_time else None
        rows.append(
            {
                "No.": idx,
                "Entry Time": entry_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "First Exit Time": first_exit_dt.strftime("%Y-%m-%d %H:%M:%S") if first_exit_dt else "",
                "Second Exit Time": second_exit_dt.strftime("%Y-%m-%d %H:%M:%S") if second_exit_dt else "",
                "Weighted PnL%": round(trade.pnl, 4),
                "Combined Score": round(getattr(trade, "combined_score", 0.0), 4),
                "Signal Type": "Trending" if getattr(trade, "is_trending_trade", False) else "Normal",
                "High (m,n)": f"({getattr(trade, 'high_m', 0):.2f}, {getattr(trade, 'high_n', 0):.2f})",
                "Low (m,n)": f"({getattr(trade, 'low_m', 0):.2f}, {getattr(trade, 'low_n', 0):.2f})",
            }
        )

    df = pd.DataFrame(rows)
    target_dir = output_dir or Path(".")
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / f"{ticker}_viphl_trade_log_{run_token}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Trade log saved to {csv_path}")
    return csv_path


def export_daily_equity_summary(
    strategy: VipHLStrategy,
    price_index: pd.DatetimeIndex,
    output_dir: Optional[Path],
    ticker: str,
    run_token: str,
):
    """Export a CSV with per-day PnL and total open position size."""
    if price_index is None or len(price_index) == 0:
        return None

    date_index = pd.DatetimeIndex(price_index)
    day_end_times = date_index.to_series().groupby(date_index.normalize()).max()
    if day_end_times.empty:
        return None

    first_timestamp = day_end_times.iloc[0]
    pnl_series = _history_to_series(
        history=getattr(strategy, "pnl_history", []),
        default_timestamp=first_timestamp,
        default_value=0.0,
    )
    position_series = _history_to_series(
        history=getattr(strategy, "position_history", []),
        default_timestamp=first_timestamp,
        default_value=0,
    )

    target_index = pd.Index(day_end_times.values, name="day_end")
    pnl_values = pnl_series.reindex(target_index, method="ffill")
    position_values = position_series.reindex(target_index, method="ffill")
    pnl_values = pnl_values.ffill().fillna(0.0)
    position_values = position_values.ffill().fillna(0).abs()

    summary_frame = pd.DataFrame(
        {
            "Date": [ts.date().isoformat() for ts in day_end_times.index],
            "Day End PnL%": pnl_values.to_numpy(dtype=float),
            "Total Position Size": position_values.astype(int).to_numpy(),
        }
    )

    target_dir = output_dir or Path(".")
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / f"{ticker}_viphl_daily_summary_{run_token}.csv"
    summary_frame.to_csv(csv_path, index=False)
    print(f"Daily equity summary saved to {csv_path}")
    return csv_path


def _history_to_series(history, default_timestamp, default_value):
    """Convert (datetime, value) pairs into a sorted Series with a default seed."""
    default_ts = pd.to_datetime(default_timestamp)
    cleaned_records = []
    for point in history or []:
        if not point:
            continue
        timestamp, value = point
        if timestamp is None:
            continue
        cleaned_records.append((pd.to_datetime(timestamp), value))
    cleaned_records.sort(key=lambda item: item[0])
    if not cleaned_records or cleaned_records[0][0] > default_ts:
        cleaned_records.insert(0, (default_ts, default_value))
    df = pd.DataFrame(cleaned_records, columns=["timestamp", "value"])
    df.sort_values("timestamp", inplace=True)
    df = df.drop_duplicates(subset="timestamp", keep="last")
    return pd.Series(df["value"].to_numpy(), index=df["timestamp"])


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    pmt = 4
    cap = 20
    parser = argparse.ArgumentParser(
        description="Run VipHLStrategy with dynamic (m, n) sizing and plot the resulting trades.",
    )
    parser.add_argument("--csv", default="BTC.csv", help="Path to the OHLCV CSV (default: %(default)s).")
    parser.add_argument("--mintick", type=float, default=0.01, help="Minimum tick size passed to the strategy.")
    parser.add_argument("--mn-start-normal", type=int, default=pmt, help="Starting m/n window for normal pivots.")
    parser.add_argument("--mn-cap-normal", type=int, default=cap, help="Maximum m/n window for normal pivots.")
    parser.add_argument("--mn-start-trend", type=int, default=pmt, help="Starting m/n window for trending pivots.")
    parser.add_argument("--mn-cap-trend", type=int, default=cap, help="Maximum m/n window for trending pivots.")
    parser.add_argument("--static-window", type=int, default=0, help="Optional override for static fallback windows.")
    parser.add_argument("--power-scaling-factor", type=float, default=1.5, help="k exponent for HL scoring.")
    parser.add_argument("--high-score-scaling-factor", type=float, default=0.5, help="Weight on high pivots.")
    parser.add_argument("--low-score-scaling-factor", type=float, default=0.5, help="Weight on low pivots.")
    parser.add_argument("--on-trend-ratio", type=float, default=1.0, help="Weight boost applied in trending mode.")
    parser.add_argument("--enable-scoring", dest="enable_scoring", action="store_true", help="Enable HL-by-point scoring.")
    parser.add_argument("--disable-scoring", dest="enable_scoring", action="store_false", help="Disable HL-by-point scoring.")
    parser.set_defaults(enable_scoring=True)
    parser.add_argument(
        "--bar-count-to-by-point",
        type=int,
        default=DEFAULT_STRATEGY_CONFIG["bar_count_to_by_point"],
        help="Override the draw-from-recent window size.",
    )
    parser.add_argument("--debug-log", default=str(RESULTS_ROOT), help="Directory or file path for debug markdown output.")
    parser.add_argument("--lookback", type=int, default=DEFAULT_STRATEGY_CONFIG["lookback"], help="Bars from the end that remain eligible for new trades.")
    parser.add_argument("--no-save", action="store_true", help="Skip saving the PNG output.")
    parser.add_argument("--show-plot", action="store_true", help="Show the matplotlib window after generation.")
    parser.add_argument("--start-date", type=str, default='2023-01-01', help="Optional inclusive start date for plotting (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default='2023-10-01', help="Optional inclusive end date for plotting (YYYY-MM-DD).")
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
        show_plot=args.show_plot,
        output_dir=output_dir,
        strategy_kwargs=strategy_kwargs,
    )


if __name__ == "__main__":
    main()
