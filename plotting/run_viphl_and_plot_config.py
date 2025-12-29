#!/usr/bin/env python3
"""Editable wrapper to launch VipHL scoring without CLI parsing."""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd
from backtrader import Cerebro

ROOT = Path(__file__).resolve().parent
RUN_SCRIPT_PATH = ROOT / "run_viphl_and_plot.py"

_spec = importlib.util.spec_from_file_location("viphl_run_entry", RUN_SCRIPT_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load VIPHL runner at {RUN_SCRIPT_PATH}")

_entry_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_entry_module)

from viphl_strategy_scoring import VipHLStrategy

run_strategy_and_plot = _entry_module.run_strategy_and_plot  # noqa: E731
build_strategy_kwargs = _entry_module.build_strategy_kwargs
resolve_csv_path = _entry_module.resolve_csv_path
DEFAULT_STRATEGY_CONFIG = _entry_module.DEFAULT_STRATEGY_CONFIG
RESULTS_ROOT = _entry_module.RESULTS_ROOT


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


DEFAULT_MAIN_OPTIONS = {
    "csv": "BTC.csv",
    "mintick": 0.01,
    "mn_start_normal": [8, 10, 12],#, 40, 50],
    "mn_cap_normal": [20, 30, 40],
    "mn_start_trend": [4,5,6],
    "mn_cap_trend": [20, 30, 40],
    "static_window": 0,
    "power_scaling_factor": 1.5,
    "high_score_scaling_factor": 0.5,
    "low_score_scaling_factor": 0.5,
    "on_trend_ratio": 1.0,
    "enable_scoring": [True, False],
    "bar_count_to_by_point": 800,#DEFAULT_STRATEGY_CONFIG["bar_count_to_by_point"],
    "debug_log": str(RESULTS_ROOT),
    "lookback": None,
    "lookback_date": '2023-09-01',
    "starting_fund": 2_000_000,
    "min_entry_size_denominator": 50,
    "no_save": False,
    "show_plot": False,
    "start_date": "2022-01-01",
    "end_date": "2025-12-01",
    "output_dir": None,
}


def _normalize_mn_values(value: Union[int, Iterable[int]]) -> List[int]:
    """Convert ints or iterables of ints into a concrete list."""
    if isinstance(value, int):
        return [value]
    if not value:
        return []
    if isinstance(value, Iterable):
        return [int(v) for v in value]
    return [int(value)]


def main(
    csv: str = DEFAULT_MAIN_OPTIONS["csv"],
    mintick: float = DEFAULT_MAIN_OPTIONS["mintick"],
    mn_start_normal: Union[int, Iterable[int]] = DEFAULT_MAIN_OPTIONS["mn_start_normal"],
    mn_cap_normal: int = DEFAULT_MAIN_OPTIONS["mn_cap_normal"],
    mn_start_trend: Union[int, Iterable[int]] = DEFAULT_MAIN_OPTIONS["mn_start_trend"],
    mn_cap_trend: int = DEFAULT_MAIN_OPTIONS["mn_cap_trend"],
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
    no_save: bool = DEFAULT_MAIN_OPTIONS["no_save"],
    show_plot: bool = DEFAULT_MAIN_OPTIONS["show_plot"],
    start_date: str = DEFAULT_MAIN_OPTIONS["start_date"],
    end_date: str = DEFAULT_MAIN_OPTIONS["end_date"],
    output_dir: Optional[str] = DEFAULT_MAIN_OPTIONS["output_dir"],
) -> Tuple[Optional[VipHLStrategy], Optional[Cerebro]]:
    """Run VipHL scoring without CLI parsing, optionally sweeping (m,n) combinations.

    Both ``mn_start_normal`` and ``mn_start_trend`` accept either single integers
    (for the standard single-run behavior) or iterables of integers. When multiple
    values are provided the wrapper performs a grid search across every combination
    and emits a summary CSV listing the parameters plus PnL, fit score, and trade count.
    """

    csv_path = resolve_csv_path(csv)
    resolved_output_dir = Path(output_dir).expanduser().resolve() if output_dir else None

    normal_values = _normalize_mn_values(mn_start_normal) or [DEFAULT_MAIN_OPTIONS["mn_start_normal"]]
    trend_values = _normalize_mn_values(mn_start_trend) or [DEFAULT_MAIN_OPTIONS["mn_start_trend"]]
    normal_cap_values = _normalize_mn_values(mn_cap_normal) or [DEFAULT_MAIN_OPTIONS["mn_cap_normal"]]
    trend_cap_values = _normalize_mn_values(mn_cap_trend) or [DEFAULT_MAIN_OPTIONS["mn_cap_trend"]]
    scoring_values = _normalize_boolean_values(enable_scoring) or [DEFAULT_MAIN_OPTIONS["enable_scoring"]]

    summary_records = []
    last_result: Tuple[Optional[VipHLStrategy], Optional[Cerebro]] = (None, None)

    for normal in normal_values:
        for trend in trend_values:
            for normal_cap in normal_cap_values:
                for trend_cap in trend_cap_values:
                    for scoring in scoring_values:
                        args = SimpleNamespace(
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
                            start_date=start_date,
                            end_date=end_date,
                        )

                        strategy_kwargs = build_strategy_kwargs(args)

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
                            }
                        )

    if summary_records:
        summary_root = resolved_output_dir or RESULTS_ROOT
        summary_root.mkdir(parents=True, exist_ok=True)
        summary_filename = f"viphl_grid_summary_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"
        summary_path = summary_root / summary_filename
        pd.DataFrame(summary_records).to_csv(summary_path, index=False)
        print(f"Grid summary saved to {summary_path}")

    return last_result


if __name__ == "__main__":
    main()
