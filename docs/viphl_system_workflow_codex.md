# VipHL Trading System – Corrected Workflow (Codex Edition)

This document mirrors the original `viphl_system_workflow.md` but fixes the inaccuracies we spotted. The key differences versus the original are:

1. **HL merge criteria** now describe the actual implementation (price overlap, bar-cross, length) instead of listing checks that do not exist (type/time overlap).  
2. **PnL computation** now reflects that percentage returns are multiplied only by the score-derived PnL scale (entry size does not change Total PnL% / Fit Score).  
3. Minor phrasing cleanups to reinforce the streaming model and the `draw_from_recent` filter.

---

## System Overview

VipHL (Very Important Pivot High/Low) is a streaming Backtrader strategy that:

1. Detects pivot highs/lows with dynamic `(m,n)` windows.
2. Converts confirmed pivots into ByPoints after trend/recency filtering.
3. Builds and merges HL (horizontal level) structures from those ByPoints.
4. Scores HLs to size trades and to scale their PnL contributions.
5. Manages break/recover windows to trigger and exit trades.

## 1. Streaming Pivot Detection

### Candidate creation
- At `current_bar`, the detector inspects `candidate_bar = current_bar - mn_start` (bounded by `mn_cap`).  
- If that bar is the extremum in `[candidate_bar - mn_start : candidate_bar + mn_start]`, it becomes an “active candidate”.

### Dynamic expansion
- Each new bar increments `current_bar` and tries `next_mn = candidate.mn + 1`.  
- As long as the candidate still holds the extremum inside the expanded window, the detector logs `candidate_expanded` and keeps tracking that same `candidate_bar`.  
- Expansion halts when the candidate loses extremum status, hits the cap, or hits a boundary.

### Confirmation / rejection
- When expansion stops, `_emit_candidate` immediately confirms the pivot using its last successful `(m,n)`.  
- The pivot series outputs the value right away; there is no batch pass “after the end of data”.

## 2. ByPoint Creation & Filters

Once a pivot series emits a confirmed point, `VipHL.add_new_by_points_to_pending` turns it into a ByPoint if it passes:

1. **Trend alignment** – e.g., only take “trending” pivots if MA trend flags match the expectation.
2. **Recency (`draw_from_recent`)** – with the default `bar_count_to_by_point=800` and `draw_from_recent=True`, the system ignores candidates older than 800 bars relative to the current bar. This is why ByPoints only appear near the latest bars in long datasets.
3. **Debug filters** – optional price/time ranges for targeted runs.

A ByPoint stores the pivot’s price, bar index/time, `(n,m)`, trend flag, and context (close-average percent, etc.).

## 3. HL Construction & Merging

### HL creation
Each accepted ByPoint instantiates an `HL` whose start/end bar/time equals the ByPoint’s pivot. The ByPoint’s `(m,n)` become the first entries in the HL’s weighted lists.

### Merge criteria (actual implementation)
For every new HL, VipHL searches existing HLs and merges if:

1. **Price overlap** (`HL.overlap`) – the HL values fall within each other’s threshold band (based on `hl_overlap_ca_percent_multiplier * close_avg_percent`). There is no separate “same type” or “time overlap” gate in the code.
2. **Bar-cross constraint** – `is_hl_bar_crosses_violated_if_merged` simulates the combined HL and ensures the number of crosses stays below `bar_cross_threshold`.
3. **Length constraint** – `is_hl_length_passed_if_merged` enforces `hl_length_threshold` (default 300 bars).

When a merge occurs, the HL’s value and `(m,n)` arrays are recalculated with weights (`last=3, second_last=2, others=1`), and debug logs output `HL merged` with `{bar_index}/{timestamp}`.

## 4. Recovery Windows & Signals

VipHL monitors each HL for break-and-recover events:

1. Detect a break (price crossing the HL by the configured body/ohlc rule).  
2. Start a recovery window: price must reclaim the HL within `trap_recover_window_threshold`.  
3. Evaluate thresholds (close above HL by `%`, ByPoint counts, etc.) to mark `is_hl_satisfied` (normal) or `is_vvip_signal` (VIP).

## 5. Scoring, Position Size, and PnL Scaling

### Scoring
`build_scoring_params` pulls the weighted `(m,n)` from the triggering HL.  
`calculate_hl_byp_score` normalizes each `(m,n)` into `0‒1`, applying the `power_scaling_factor` and the high/low scaling factors.  
`combined_score` averages the weighted high/low scores and drives:

- **Entry size:** `base_size = floor(order_size_usd / price)` → `entry_size = max(1, floor(base_size * combined_score))`. This affects only the trade log / plot markers.
- **PnL scale:** `scale = 1.0 + 2.0 * combined_score`. The scale is stored per trade in `trade_scales`.

### PnL application (corrected)
- When a leg exits, `cur_return` (percentage) is multiplied by the stored scale:  
  - First leg: `delta_pnl = cur_return * scale`.  
  - Second leg (if any): `delta_pnl = cur_return * (2/3) * scale` to reflect the smaller remaining size.  
- There is **no multiplication by entry_size** in the strategy’s PnL math; Total Pnl% and Fit Score are sums of these scaled percentage deltas, not dollarized exposures.

## 6. Trade Management

Trades monitor:
- Stop loss (`min(last two lows)`)
- Take-profit ladder (partial exit after volatility-based gains, optional second leg)
- Cycle timeout (e.g., 6 months)
- Drawdown tracking and final statistics (Total PnL%, Avg Winner%, Fit Score). Fit Score = `min(-win_pnl / loss_pnl, 500)` when losses exist; otherwise capped at 500.

## File Pointers
- Pivot detectors: `viphl-source-code/indicators/helper/pivot_high.py`, `pivot_low.py`
- ByPoint/HL handling: `viphl-source-code/indicators/viphl/dto/viphl.py`
- Strategy + scoring/PnL: `hl-scroing/viphl_strategy_scoring.py`
- CLI runner + plotting: `hl-scroing/run_viphl_and_plot.py`


  1. Streaming model description — accurate.
     PivotLow.next/PivotHigh.next consume one bar at a time, seed new candidates at
     current_index - mn_start, and immediately start expanding them as new bars arrive.
     There’s no batch post-processing; confirmation happens as soon as the candidate
      - Trend filters: add_new_by_points_to_pending enforces expected trend flags
        before logging a ByPoint.
      - Recency filter: only applies if Settings.draw_from_recent is True; under
        that default, pivots are ignored until last_bar_index - bar_index <
        bar_count_to_by_point, which explains why ByPoints cluster near the most recent
        800 bars.
      - Debug ranges: price/time filters short-circuit the ByPoint if the user set
        them. citeturn1view0
  4. HL creation/merging — partially inaccurate.
      - Correct: every new ByPoint spawns an HL, and merges reuse weighted averaging
        with the last/second_last/by_point weights.
      - Incorrect statements:
          - “Same Type Check” and “Time-overlap check” are not implemented—VipHL
            only tests price overlap via HL.overlap, so highs/lows aren’t explicitly
            separated and time doesn’t factor into the merge guard.
          - “Bar cross check < threshold” is enforced, but the text should mention
            it’s done via is_hl_bar_crosses_violated_if_merged, not a simple counter
            compare.
            Please update those bullets so they describe the actual price-threshold
            overlap + bar-cross + length checks coded at viphl-source-code/indicators/
            viphl/dto/viphl.py:336-404. citeturn1view0
  5. Scoring and sizing — mostly accurate, with one key mistake.
      - Correct: build_scoring_params pulls the weighted (m,n) from the HL,
        compute_scoring_metrics normalizes them, and entry_size = floor(base_entry_size
        * combined_score) (min 1) matches the doc. citeturn2fetch0
      - Incorrect: the document says “final_pnl = percent_return * entry_size *
        pnl_scale.” In reality, exit_first_time/exit_second_time add percent_return *
        scale (or * 2/3 * scale) to total_pnl, independent of entry_size. Entry size
        only affects how many shares are notionally opened (and what the plot shows);
        PnL% and Fit Score are purely driven by scaled percentage returns. Please
        adjust that formula to reflect final_pnl = percent_return * pnl_scale (second
        leg uses 2/3). citeturn2fetch0
  6. PnL scaling explanation — accurate where it matches the code.
     scale = 1 + 2 * combined_score is exactly what calculate_pnl_scale does, and it’s
     stored per-trade for later exits. citeturn2fetch0

