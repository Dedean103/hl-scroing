## Dynamic m/n Detection – Implementation Plan

### Goals
- Replace static `(m,n)` pivot windows with a progressive detector that expands the window until failure or a configurable cap (default 20).
- Preserve the downstream HL/Recovery/Trade logic by emitting `ByPoint`s with accurate `(m,n)` metadata.

### Repo snapshot (from `CLAUDE.md`) – all present already
- `VipHLStrategy` and its Trade/Recovery/Hl scoring ecosystem already live under `hl-scroing/viphl_strategy_scoring.py` and `viphl-source-code/viphl_strategy_simplfied.py`, so no architectural changes are needed before dynamic `(m,n)` work.
- The helper indicators (`indicators/helper/pivot_high.py`, `pivot_low.py`, `close_average.py`, `percentile_nearest_rank.py`) and DTOs (`dto/trade_v2.py`, `viphl/dto/*`) already implement the pieces referenced by the guidelines.
- CSV feeding, plotting utilities, and TradingView integration are already part of the repo structure; our new runner (`hl-scroing/run_viphl_and_plot.py`) ties everything together with a single Backtrader entry point.
- Since the foundational components exist, we only need to layer the dynamic-detection instrumentation, logging, and documentation described in the plan.

### CLAUDE plan alignment
- **Phase 1 (current_bar logging)** – implemented by `VipHL._log_debug_markdown` writing `current_bar` for every ByPoint/HL event, plus the Markdown trace; already done.
- **Phase 2 (pivot indicator instrumentation)** – new `PivotHigh`/`PivotLow` logging hooks now capture candidate lifecycle events; implemented in `viphl-source-code/indicators/helper/pivot_{high,low}.py`.
- **Phase 3 (HL construction logging)** – HL creation/merge events already log to Markdown; covered in `viphl-source-code/indicators/viphl/dto/viphl.py`.
- **Phase 4 (HL lifecycle tracking)** – still pending, as the existing plan only logs creation/merge; future work can extend this to invalidation/recovery windows/trade signals.

### Step-by-step plan

1. **Parameter scaffolding** ✅ *(completed)*
   1. Added strategy params (`mn_start_point_*`, `mn_cap_*`) for normal + trending, highs + lows (same defaults for now).
   2. Threaded the new params into `Settings` so downstream components can consume them.
   3. Docs updated via `discussion_summary.md`; next doc refresh will mention these knobs explicitly.

2. **Enhanced pivot detectors** ✅ *(completed)*
   1. Create new indicators (or extend `PivotHigh`/`PivotLow`) that:
      - Maintain a list/dict of “live” pivot candidates keyed by bar index.
      - For each candidate, store current `(m,n)` and whether it has already been emitted.
      - On every `next()` call:
        - Attempt to validate each live candidate using `(m+1, n+1)` if enough bars exist.
        - If validation passes, update the candidate’s `(m,n)` and wait for the next bar.
        - If validation fails or `m` hits `mn_cap`, emit the pivot with the last successful `(m,n)` and remove it from the live set.
      - When the static base condition first triggers (at `mn_start_point`), add the candidate to the live set with `(m,n)=mn_start_point`.
   2. Edge cases:
      - If `bar_index - n < 0` or `bar_index + m >= len(data)` during validation, stop expanding and emit immediately.
      - Ensure trending vs. normal indicators pull the correct start/cap parameters.

3. **ByPoint / HL metadata adjustments** ✅ *(completed)*
   1. Extend `ByPoint` dataclass to include `n` and `m`.
   2. Update `VipHL.add_new_by_points_to_pending` to capture the dynamic `(n,m)` from the indicator (probably via `self.normal_high_by_point.current_n`, etc.).
   3. Update `HL` (if helpful) to store per-pivot `(n,m)` sequences—useful for future scoring or debugging.

4. **Scoring & weighting integration** ✅ *(completed)*
   1. Used the dynamic `(m,n)` from the HL that triggered the recovery window (Option B) by computing weighted `(m,n)` via the existing `last/second_last/by_point` scheme.
   2. Fed those values into `calculate_hl_byp_score` for both strategies’ `quote_trade`/`record_trade`, so position sizing and PnL scaling reflect actual pivot depth.
   3. Hooks remain for future overlap/weight tweaks if we want to emphasize deep pivots even more.

5. **Testing & validation**
   1. Add unit-like tests (could be simple scripts) that feed deterministic OHLC data to the new pivot detectors and assert:
      - Pivots expand from start to cap when the extremum remains intact.
      - Expansion stops when a higher/lower bar appears or history bounds are reached.
   2. Run the Backtrader strategy on sample data to confirm HL counts, recovery windows, and trade stats remain sensible.
   3. Update the README/discussion docs with instructions on tuning the new parameters.
   4. ✅ *Completed 2025‑12‑01*: Delivered `hl-scroing/run_viphl_and_plot.py`, a single runner that executes the dynamic `(m,n)` strategy, prints the key stats, and generates the trade/HL plots—fully replacing the legacy `plotting/` scripts so future docs/regression captures all use the same entry point.

6. **Future enhancements (not in first pass but documented)**
   - Separate start/cap tuning for highs vs. lows, normal vs. trending.
   - Visualization/logging of pivot upgrade events for debugging.
   - Optionally, allow asymmetric expansion (e.g., increase `m` faster than `n` in trending markets).
   - *(Done as part of logging work)* Debugging hooks now write ByPoint/HL events to a Markdown log when `debug_mode` and `debug_log_path` are set.

This document can serve as the checklist when we start coding the dynamic behavior. Adjust or append steps as requirements evolve.
