#!/usr/bin/env python3
"""
Run VipHL strategy with dynamic (m, n) detector and visualize results.

Supports both single-run and grid-search modes:
- Single run: main(mn_start_normal=10, mn_start_trend=4)
- Grid search: main(mn_start_normal=[8,10,12], mn_start_trend=[4,5,6])
- Direct execution: python run_viphl_and_plot.py (uses DEFAULT_MAIN_OPTIONS)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import backtrader as bt
from backtrader import num2date
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from types import SimpleNamespace


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
    "starting_fund": 2_000_000,
    "min_entry_size_denominator": 100,
    "risk_free_rate": 0.0,  # Annual risk-free rate (e.g., 0.04 for 4%)
}

DEFAULT_MAIN_OPTIONS = {
    "csv": "BTC.csv",
    "mintick": 0.01,
    "mn_start_normal": [8,10],#[8, 10, 12],
    "mn_cap_normal": 20,#[20, 30, 40],
    "mn_start_trend": 4,#[4, 5, 6],
    "mn_cap_trend": 20,#[20, 30, 40],
    "static_window": 0,
    "power_scaling_factor": 1.5, #k
    "high_score_scaling_factor": 0.5,
    "low_score_scaling_factor": 0.5,
    "on_trend_ratio": 1.0,
    "enable_scoring": True, #[True, False],
    "bar_count_to_by_point": 800,
    "debug_log": str(RESULTS_ROOT),
    "lookback": None,
    "lookback_date": "2023-09-01",
    "starting_fund": 2_000_000,
    "min_entry_size_denominator": 50,
    "risk_free_rate": 0.0,  # Annual risk-free rate (e.g., 0.04 for 4%)
    "no_save": False,
    "show_plot": False,
    "start_date": "2022-01-01",
    "end_date": "2025-12-01",
    "output_dir": None,
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


def _normalize_mn_values(value: Union[int, Iterable[int]]) -> List[int]:
    """Convert ints or iterables of ints into a concrete list."""
    if isinstance(value, int):
        return [value]
    if not value:
        return []
    if isinstance(value, Iterable):
        return [int(v) for v in value]
    return [int(value)]


def _normalize_boolean_values(value: Union[bool, Iterable[bool]]) -> List[bool]:
    """Turn a bool/iterable into a normalized list of bools."""
    true_values = {"1", "true", "t", "yes", "y"}
    if isinstance(value, bool):
        return [value]
    if isinstance(value, str):
        return [value.lower() in true_values]
    if not value:
        return []
    if isinstance(value, Iterable):
        normalized = []
        for entry in value:
            if isinstance(entry, bool):
                normalized.append(entry)
            else:
                normalized.append(str(entry).lower() in true_values)
        return normalized
    return [bool(value)]


def build_strategy_kwargs(
    csv: str,
    mintick: float,
    mn_start_normal: int,
    mn_cap_normal: int,
    mn_start_trend: int,
    mn_cap_trend: int,
    static_window: int,
    power_scaling_factor: float,
    high_score_scaling_factor: float,
    low_score_scaling_factor: float,
    on_trend_ratio: float,
    enable_scoring: bool,
    bar_count_to_by_point: int,
    debug_log: str,
    lookback: int,
    lookback_date: Optional[str],
    starting_fund: float,
    min_entry_size_denominator: float,
    start_date: str,
    end_date: str,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """Merge configuration values into the default strategy configuration."""
    config = dict(DEFAULT_STRATEGY_CONFIG)
    config["mintick"] = mintick
    config["lookback"] = lookback

    config["mn_start_point_high"] = mn_start_normal
    config["mn_start_point_low"] = mn_start_normal
    config["mn_cap_high"] = mn_cap_normal
    config["mn_cap_low"] = mn_cap_normal

    config["mn_start_point_high_trend"] = mn_start_trend
    config["mn_start_point_low_trend"] = mn_start_trend
    config["mn_cap_high_trend"] = mn_cap_trend
    config["mn_cap_low_trend"] = mn_cap_trend

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
        config[key] = static_window if static_window else config[key]

    config["power_scaling_factor"] = power_scaling_factor
    config["high_score_scaling_factor"] = high_score_scaling_factor
    config["low_score_scaling_factor"] = low_score_scaling_factor
    config["on_trend_ratio"] = on_trend_ratio
    if debug_log:
        config["debug_log_path"] = debug_log
    config["enable_hl_byp_scoring"] = bool(enable_scoring)
    config["bar_count_to_by_point"] = bar_count_to_by_point
    if start_date:
        config["plot_start_date"] = start_date
    if end_date:
        config["plot_end_date"] = end_date
    config["starting_fund"] = starting_fund
    config["min_entry_size_denominator"] = min_entry_size_denominator
    config["risk_free_rate"] = risk_free_rate
    if lookback_date:
        config["lookback_date"] = lookback_date

    return config


def _compute_lookback_from_date(index: pd.Index, lookback_date: Optional[str]) -> Optional[int]:
    """Convert a date string into bar-count lookback relative to the dataset end."""
    if not lookback_date:
        return None
    try:
        target_date = pd.to_datetime(lookback_date)
    except (TypeError, ValueError):
        return None
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return None
    last_pos = len(index) - 1
    try:
        first_allowed = next(i for i, ts in enumerate(index) if ts >= target_date)
    except StopIteration:
        return None
    return max(0, last_pos - first_allowed)


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
    lookback_date = strategy_args.pop("lookback_date", None)
    if lookback_date:
        computed_lookback = _compute_lookback_from_date(dataframe.index, lookback_date)
        if computed_lookback is not None:
            strategy_args["lookback"] = computed_lookback

    run_token = "{normal}_{normal_cap}_{trend}_{trend_cap}_{scoreflag}_bar{bar_count}_{stamp}".format(
        normal=strategy_args.get("mn_start_point_high", "na"),
        trend=strategy_args.get("mn_start_point_high_trend", "na"),
        normal_cap=strategy_args.get("mn_cap_high", "na"),
        trend_cap=strategy_args.get("mn_cap_high_trend", "na"),
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
        pnl_plot_filename = output_root / f"{ticker}_cumu_pnl_{run_token}.png"
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

    # Generate BTC price overlay charts if daily summary was created
    if save_plot:
        daily_summary_csv = output_root / f"{ticker}_daily_summary_{run_token}.csv"
        btc_csv = csv_file.parent / "BTC.csv"
        if daily_summary_csv.exists() and btc_csv.exists():
            generate_btc_overlay_charts(
                summary_csv=daily_summary_csv,
                btc_csv=btc_csv,
                output_dir=output_root,
                run_token=run_token,
                ticker=ticker,
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
        f"Sharpe Ratio: {result_stats.get('Sharpe Ratio', 0):.3f}\n"
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
        high_source = getattr(trade, "high_source", "") or ""
        low_source = getattr(trade, "low_source", "") or ""
        static_high_used = high_source != "dynamic" and bool(high_source)
        static_low_used = low_source != "dynamic" and bool(low_source)
        rows.append(
            {
                "No.": idx,
                "Entry Time": entry_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "First Exit Time": first_exit_dt.strftime("%Y-%m-%d %H:%M:%S") if first_exit_dt else "",
                "Second Exit Time": second_exit_dt.strftime("%Y-%m-%d %H:%M:%S") if second_exit_dt else "",
                "Weighted PnL%": round(trade.pnl, 4),
                "Combined Score": round(getattr(trade, "combined_score", 0.0), 4),
                "Signal Type": "Trending" if getattr(trade, "is_trending_trade", False) else "Normal",
                "Static High Used": static_high_used,
                "Static Low Used": static_low_used,
                "High (m,n)": f"({getattr(trade, 'high_m', 0):.2f}, {getattr(trade, 'high_n', 0):.2f})",
                "Low (m,n)": f"({getattr(trade, 'low_m', 0):.2f}, {getattr(trade, 'low_n', 0):.2f})",
            }
        )

    df = pd.DataFrame(rows)
    target_dir = output_dir or Path(".")
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / f"{ticker}_trade_log_{run_token}.csv"
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
    fund_series = _history_to_series(
        history=getattr(strategy, "remaining_fund_history", []),
        default_timestamp=first_timestamp,
        default_value=getattr(strategy, "starting_fund", 0.0),
    )

    target_index = pd.Index(day_end_times.values, name="day_end")
    pnl_values = pnl_series.reindex(target_index, method="ffill")
    position_values = position_series.reindex(target_index, method="ffill")
    fund_values = fund_series.reindex(target_index, method="ffill")
    pnl_values = pnl_values.ffill().fillna(0.0)
    position_values = position_values.ffill().fillna(0).abs()
    fund_values = fund_values.ffill().fillna(getattr(strategy, "starting_fund", 0.0))
    prev_fund = fund_values.shift(1)
    daily_change_values = fund_values.subtract(prev_fund).div(prev_fund.replace(0, pd.NA)).mul(100).fillna(0.0)

    summary_frame = pd.DataFrame(
        {
            "Date": [ts.date().isoformat() for ts in day_end_times.index],
            "Day End PnL%": pnl_values.to_numpy(dtype=float),
            "Daily Change%": daily_change_values.to_numpy(dtype=float),
            "Remaining Fund": fund_values.to_numpy(dtype=float),
        }
    )

    target_dir = output_dir or Path(".")
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / f"{ticker}_daily_summary_{run_token}.csv"
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


def _load_btc_price(price_csv: Path) -> pd.Series:
    """Load BTC daily closing prices from CSV."""
    price_csv = Path(price_csv).expanduser().resolve()
    if not price_csv.is_file():
        return pd.Series(dtype=float)

    try:
        price_df = pd.read_csv(price_csv, parse_dates=["datetime"])
        if "close" not in price_df.columns:
            return pd.Series(dtype=float)

        price_df["Date"] = price_df["datetime"].dt.normalize()
        price_df.sort_values("Date", inplace=True)
        daily_price = price_df.groupby("Date")["close"].last()
        daily_price.name = "BTC Close"
        return daily_price
    except Exception:
        return pd.Series(dtype=float)


def _align_price(summary_dates: pd.Series, price_series: pd.Series) -> pd.Series:
    """Align BTC price data to summary dates using forward/backward fill."""
    if price_series.empty:
        return pd.Series(dtype=float)
    date_index = pd.DatetimeIndex(summary_dates)
    aligned = price_series.reindex(date_index)
    aligned = aligned.ffill().bfill()
    return pd.Series(aligned.to_numpy(), index=date_index, name="BTC Close")


def _plot_remaining_fund(summary_df: pd.DataFrame, price_series: pd.Series, output_path: Path, show_plot: bool) -> Path:
    """Plot remaining fund (portfolio value) with BTC price overlay."""
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
    print(f"Remaining fund plot saved to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def _plot_day_end_pnl(summary_df: pd.DataFrame, price_series: pd.Series, output_path: Path, show_plot: bool) -> Path:
    """Plot day-end PnL% with BTC price overlay."""
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

    ax_left.set_title("Day End PnL% vs BTC Price")
    fig.autofmt_xdate()

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Day-end PnL plot saved to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def _plot_daily_change_bars(summary_df: pd.DataFrame, price_series: pd.Series, output_path: Path, show_plot: bool) -> Path:
    """Plot daily change % bars with BTC price overlay."""
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

    ax_left.set_title("Daily Increment vs BTC Price")
    fig.autofmt_xdate()

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Daily increment plot saved to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def generate_btc_overlay_charts(
    summary_csv: Path,
    btc_csv: Path,
    output_dir: Path,
    run_token: str,
    ticker: str,
    show_plot: bool = False,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Generate 3 BTC overlay charts from daily summary CSV."""
    try:
        # Load data
        summary_df = pd.read_csv(summary_csv, parse_dates=["Date"])
        if summary_df.empty:
            return None, None, None

        price_series = _load_btc_price(btc_csv)
        if price_series.empty:
            return None, None, None

        aligned_price = _align_price(summary_df["Date"], price_series)

        # Generate charts
        remaining_path = output_dir / f"{ticker}_remaining_fund_{run_token}.png"
        day_end_path = output_dir / f"{ticker}_day_end_pnl_{run_token}.png"
        daily_path = output_dir / f"{ticker}_daily_increment_{run_token}.png"

        _plot_remaining_fund(summary_df, aligned_price, remaining_path, show_plot)
        _plot_day_end_pnl(summary_df, aligned_price, day_end_path, show_plot)
        _plot_daily_change_bars(summary_df, aligned_price, daily_path, show_plot)

        return remaining_path, day_end_path, daily_path
    except Exception as e:
        print(f"Error generating BTC overlay charts: {e}")
        return None, None, None


def main(
    csv: str = DEFAULT_MAIN_OPTIONS["csv"],
    mintick: float = DEFAULT_MAIN_OPTIONS["mintick"],
    mn_start_normal: Union[int, Iterable[int]] = DEFAULT_MAIN_OPTIONS["mn_start_normal"],
    mn_cap_normal: Union[int, Iterable[int]] = DEFAULT_MAIN_OPTIONS["mn_cap_normal"],
    mn_start_trend: Union[int, Iterable[int]] = DEFAULT_MAIN_OPTIONS["mn_start_trend"],
    mn_cap_trend: Union[int, Iterable[int]] = DEFAULT_MAIN_OPTIONS["mn_cap_trend"],
    static_window: int = DEFAULT_MAIN_OPTIONS["static_window"],
    power_scaling_factor: float = DEFAULT_MAIN_OPTIONS["power_scaling_factor"],
    high_score_scaling_factor: float = DEFAULT_MAIN_OPTIONS["high_score_scaling_factor"],
    low_score_scaling_factor: float = DEFAULT_MAIN_OPTIONS["low_score_scaling_factor"],
    on_trend_ratio: float = DEFAULT_MAIN_OPTIONS["on_trend_ratio"],
    enable_scoring: Union[bool, Iterable[bool]] = DEFAULT_MAIN_OPTIONS["enable_scoring"],
    bar_count_to_by_point: int = DEFAULT_MAIN_OPTIONS["bar_count_to_by_point"],
    debug_log: str = DEFAULT_MAIN_OPTIONS["debug_log"],
    lookback: int = DEFAULT_MAIN_OPTIONS["lookback"],
    lookback_date: Optional[str] = DEFAULT_MAIN_OPTIONS["lookback_date"],
    starting_fund: float = DEFAULT_MAIN_OPTIONS["starting_fund"],
    min_entry_size_denominator: float = DEFAULT_MAIN_OPTIONS["min_entry_size_denominator"],
    risk_free_rate: float = DEFAULT_MAIN_OPTIONS["risk_free_rate"],
    no_save: bool = DEFAULT_MAIN_OPTIONS["no_save"],
    show_plot: bool = DEFAULT_MAIN_OPTIONS["show_plot"],
    start_date: str = DEFAULT_MAIN_OPTIONS["start_date"],
    end_date: str = DEFAULT_MAIN_OPTIONS["end_date"],
    output_dir: Optional[str] = DEFAULT_MAIN_OPTIONS["output_dir"],
) -> Tuple[Optional[VipHLStrategy], Optional[Cerebro]]:
    """Run VipHL strategy with support for grid search.

    Accepts either single values or iterables of values for parameters.
    When iterables are provided, runs a grid search across all combinations.

    Example single run:
        main(mn_start_normal=10, mn_start_trend=4)

    Example grid search:
        main(
            mn_start_normal=[8, 10, 12],
            mn_start_trend=[4, 5, 6],
            mn_cap_normal=[20, 30]
        )
    """

    csv_path = resolve_csv_path(csv)
    resolved_output_dir = Path(output_dir).expanduser().resolve() if output_dir else None

    normal_values = _normalize_mn_values(mn_start_normal)
    trend_values = _normalize_mn_values(mn_start_trend)
    normal_cap_values = _normalize_mn_values(mn_cap_normal)
    trend_cap_values = _normalize_mn_values(mn_cap_trend)
    scoring_values = _normalize_boolean_values(enable_scoring)

    summary_records = []
    last_result: Tuple[Optional[VipHLStrategy], Optional[Cerebro]] = (None, None)

    for normal in normal_values:
        for trend in trend_values:
            for normal_cap in normal_cap_values:
                for trend_cap in trend_cap_values:
                    for scoring in scoring_values:
                        strategy_kwargs = build_strategy_kwargs(
                            csv=csv,
                            mintick=mintick,
                            mn_start_normal=normal,
                            mn_cap_normal=normal_cap,
                            mn_start_trend=trend,
                            mn_cap_trend=trend_cap,
                            static_window=static_window,
                            power_scaling_factor=power_scaling_factor,
                            high_score_scaling_factor=high_score_scaling_factor,
                            low_score_scaling_factor=low_score_scaling_factor,
                            on_trend_ratio=on_trend_ratio,
                            enable_scoring=scoring,
                            bar_count_to_by_point=bar_count_to_by_point,
                            debug_log=debug_log,
                            lookback=lookback,
                            lookback_date=lookback_date,
                            starting_fund=starting_fund,
                            min_entry_size_denominator=min_entry_size_denominator,
                            risk_free_rate=risk_free_rate,
                            start_date=start_date,
                            end_date=end_date,
                        )

                        is_grid_search = len(normal_values) > 1 or len(trend_values) > 1 or len(normal_cap_values) > 1 or len(trend_cap_values) > 1 or len(scoring_values) > 1
                        if is_grid_search:
                            print(
                                f"=== Running grid combination: normal {normal}, trend {trend}, normal_cap {normal_cap}, "
                                f"trend_cap {trend_cap}, scoring {scoring} ==="
                            )

                        strat, cerebro = run_strategy_and_plot(
                            csv_file=csv_path,
                            save_plot=not no_save,
                            show_plot=show_plot,
                            output_dir=resolved_output_dir,
                            strategy_kwargs=strategy_kwargs,
                        )
                        last_result = (strat, cerebro)

                        stats = getattr(strat, "result", {})
                        summary_records.append(
                            {
                                "mn_start_normal": normal,
                                "mn_cap_normal": normal_cap,
                                "mn_start_trend": trend,
                                "mn_cap_trend": trend_cap,
                                "enable_scoring": scoring,
                                "Total Pnl%": stats.get("Total Pnl%", 0.0),
                                "Fit Score": stats.get("Fit Score", 0.0),
                                "Trade Count": stats.get("Trade Count", 0),
                                "Scale": stats.get("Scale", 0),
                                "Sharpe Ratio": stats.get("Sharpe Ratio", 0.0),
                            }
                        )

    if summary_records and len(summary_records) > 1:
        summary_root = resolved_output_dir or RESULTS_ROOT
        summary_root.mkdir(parents=True, exist_ok=True)
        summary_filename = f"grid_summary_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"
        summary_path = summary_root / summary_filename
        pd.DataFrame(summary_records).to_csv(summary_path, index=False)
        print(f"Grid summary saved to {summary_path}")

    return last_result


if __name__ == "__main__":
    main()
