# PnL Scaling Implementation Summary

## Overview
Implemented a PnL scaling system that replaces position sizing adjustment with a score-based PnL multiplier. Each trade's PnL is now scaled by a factor (1-3) based on the combined score at entry time.

## Changes Made

### 1. New Function: `calculate_pnl_scale()` (Line 137-154)
```python
def calculate_pnl_scale(self, combined_score):
    '''
    Maps combined_score (0-1) to PnL scale (1-3) using exponential curve
    Formula: scale = 1 + 2 * (score ^ exponent)
    '''
    exponent = 2.0
    scale = 1.0 + 2.0 * (combined_score ** exponent)
    return scale
```

**Scale Mapping (Quadratic with exponent=2):**
- Score 0.0 → Scale 1.0 (minimum)
- Score 0.5 → Scale 1.5 (mid-range)
- Score 0.75 → Scale 2.125
- Score 1.0 → Scale 3.0 (maximum)

### 2. Trade Scale Storage (Line 240)
Added dictionary to track PnL scale for each trade:
```python
self.trade_scales = {}  # Maps trade id to PnL scale (1-3)
```

### 3. Modified `record_trade()` (Lines 384-447)
**Removed position sizing adjustment:**
```python
# OLD: entry_size = math.floor(base_entry_size * combined_score)
# NEW: entry_size = base_entry_size
```

**Added scale calculation and storage:**
```python
pnl_scale = self.calculate_pnl_scale(combined_score)
self.trade_scales[id(self.trade_list[-1])] = pnl_scale
```

### 4. Modified PnL Calculations
All PnL calculations now apply the scale multiplier:

**Partial Exit (Lines 507-510):**
```python
scale = self.trade_scales.get(id(trade), 1.0)
self.total_pnl += cur_return / 3 * scale
trade.pnl += cur_return / 3 * scale
```

**exit_first_time() (Lines 530-533):**
```python
scale = self.trade_scales.get(id(trade), 1.0)
self.total_pnl += cur_return * scale
trade.pnl += cur_return * scale
```

**exit_second_time() (Lines 540-543):**
```python
scale = self.trade_scales.get(id(trade), 1.0)
self.total_pnl += cur_return * 2 / 3 * scale
trade.pnl += cur_return * 2 / 3 * scale
```

## Key Behaviors

### Before vs After
| Aspect | Before | After |
|--------|--------|-------|
| Position Size | Variable (score × base_size) | Fixed (base_size) |
| PnL Calculation | Return × 1.0 | Return × scale (1-3) |
| High Score Impact | Larger position | Higher PnL multiplier |
| Low Score Impact | Smaller position | Lower PnL multiplier |

### Example Scenario
**Trade with combined_score = 0.5:**
- Scale = 1 + 2 × (0.5²) = 1.5
- Base entry size = 40 BTC
- Entry size used = 40 BTC (not 20 BTC anymore)
- If trade returns +10%:
  - PnL recorded = 10% × 1.5 = 15%

**Trade with combined_score = 1.0:**
- Scale = 1 + 2 × (1.0²) = 3.0
- Entry size used = 40 BTC
- If trade returns +10%:
  - PnL recorded = 10% × 3.0 = 30%

## Testing
Run with debug mode to see scale calculations:
```bash
python test_pnl_scale.py
```

## Impact on Fit Score
The fit score automatically reflects scaled PnL since it's calculated from `trade.pnl` values:
```python
fit_score = min(-win_pnl / loss_pnl, FIT_SCORE_MAX)
```

Higher quality trades (higher scores) now contribute more to both winning and losing PnL calculations.

## Customization
To adjust the curve steepness, modify the exponent in `calculate_pnl_scale()`:
- `exponent = 1.0`: Linear mapping (score 0.5 → scale 2.0)
- `exponent = 2.0`: Quadratic (score 0.5 → scale 1.5) **[CURRENT]**
- `exponent = 3.0`: Cubic (score 0.5 → scale 1.25, more aggressive)
