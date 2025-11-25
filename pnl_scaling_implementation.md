# Position Scaling Implementation Summary

## Overview
The VIPHL strategy now uses the combined HL score (derived from the dynamic m/n windows) to scale actual position size directly. The score is mapped to a multiplier between **1× and 3×** of the base order size so higher-confidence pivots risk more capital while weaker setups stay small. PnL is recorded at face value using the sized position—there is no additional score-based weighting layer anymore.

## Key Mechanics
- **Scaling helper**: `calculate_position_scale(score)` returns `1 + 2 * score` and clamps the result to `[1, 3]`.
- **Entry sizing**: both `quote_trade` and `record_trade` compute `entry_size = floor(base_size * scale)`, where `base_size = floor(order_size_in_usd / close)`.
- **PnL accounting**: exits simply add/subtract the realized percentage return (or fractions for partial exits). Because the trade carried the larger/smaller size already, no extra multiplier is required.
- **Visualization**: marker sizes in `trade_visualization.py` now reflect each trade's `total_entry_size` relative to the smallest trade in the sample, so you can still see which signals were sized up.

## Rationale
Previously the system used a fixed position size and multiplied PnL by a score-based factor. That double-counted conviction when combined with dynamic sizing, inflating both wins and losses. Removing the PnL multiplier keeps backtest metrics (`Total Pnl%`, `Fit Score`, etc.) aligned with what the scaled trades actually produced.

## Extending
If you need a different exposure curve, adjust the helper:
```python
scale = 1.0 + k * combined_score
scale = max(min_scale, min(scale, max_scale))
```
where `k`, `min_scale`, and `max_scale` define your preferred slope and bounds.
