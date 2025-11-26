# VipHL Mechanisms Summary (Version 2025-11-26)

## HL Construction
- Every HL originates from a `ByPoint` created by the pivot detectors inside `VipHL.add_new_by_points_to_pending` (`viphl-source-code/indicators/viphl/dto/viphl.py`).
- `rebuild_hl_from_most_recent_by_point` rebuilds the HL list whenever fresh byPoints exist, instantiating an `HL` with the pivot price/index and then attempting to merge it into earlier HLs (`viphl-source-code/indicators/viphl/dto/viphl.py`).
- HL merging (`HL.merge`) concatenates the contributing `by_point_values`, recalculates the weighted price via `calculate_weighted_hl_value`, and updates the line’s span/metadata while ensuring overlap, bar-cross, and length constraints remain satisfied (`viphl-source-code/indicators/viphl/dto/hl.py`).
- After construction, `extend_hl_to_first_cross_from_history`/`extend_hl_to_first_cross` extend each HL to the first valid cross and keep counting additional crosses so later recovery checks can enforce `hl_extend_bar_cross_threshold` (`viphl-source-code/indicators/viphl/dto/viphl.py`).

## Trade Triggering
- `update_recovery_window` monitors every HL for breaks (close below, wick break, “low above” condition) and creates `RecoveryWindow` objects that capture break price, bar index, and bar-cross stats (`viphl-source-code/indicators/viphl/dto/viphl.py`).
- `check_recovery_window_v3` marks a window as recovered only if price reclaims both the HL and the relevant lows within `trap_recover_window_threshold`, satisfies `signal_window` spacing, the minimum “bars closed above HL” count, and the `hl_extend_bar_cross_threshold`. Passing windows become VVIP when their HL merged enough byPoints (`viphl-source-code/indicators/viphl/dto/viphl.py`).
- In `VipHLStrategy.next`, a trade fires when the flattened recovery result reports `is_hl_satisfied` or `is_vvip_signal` *and* the quoted stop-loss percentage stays below the respective threshold. On trigger, `record_trade` sizes the position using the current m/n-based HL scores and logs the trade (`viphl_strategy_scoring.py`).

## Key Takeaways
- HLs are purely a byPoint aggregation mechanism; no other path can create them.
- Recovery windows form the bridge between HL structure and actual orders, enforcing timing, spacing, and cleanliness rules before any trade is recorded.
- Position sizing depends on the dynamic or static m/n windows available at trigger time, tying exposure to the detected pivot quality.
