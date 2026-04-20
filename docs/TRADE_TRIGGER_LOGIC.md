# Trade Trigger Logic Analysis

## Overview
This document explains the complete trade entry logic for the VipHL strategy and all relevant parameters.

---

## Trade Entry Flow (in `next()` method)

### Step 1: Update VipHL State
```python
self.viphl.update_built_in_vars(bar_index=self.bar_index(), last_bar_index=self.last_bar_index())
self.viphl.update(is_ma_trending=self.is_ma_trending, close_avg_percent=self.close_average_percent[0])
```
- Updates current bar index and trending state
- Calculates close average percentage

### Step 2: Update Recovery Window (站稳 = "Standing Firm")
```python
self.viphl.update_recovery_window(
    trap_recover_window_threshold=self.params.trap_recover_window_threshold,  # Default: 6 (bars)
    search_range=self.params.close_above_hl_search_range,                      # Default: 5 (bars)
    low_above_hl_threshold=self.params.low_above_hl_threshold,                 # Default: 0.5 (multiplier of CA%)
    close_avg_percent=self.close_average_percent[0]
)
```
**What it does:** Checks if price has "recovered" above a HL level and is stable

### Step 3: Check Recovery Window
```python
recovery_window_result = self.viphl.check_recovery_window_v3(
    close_avg_percent=self.close_average_percent[0],
    close_above_hl_threshold=self.params.close_above_hl_threshold,              # Default: 0.25 (multiplier of CA%)
    trap_recover_window_threshold=self.params.trap_recover_window_threshold,    # Default: 6 (bars)
    signal_window=self.params.signal_window,                                     # Default: 2 (bars)
    close_above_low_threshold=self.params.close_above_low_threshold,            # Default: 1.25 (multiplier of CA%)
    close_above_recover_low_threshold=self.params.close_above_recover_low_threshold,  # Default: 1.25 (multiplier of CA%)
    bar_count_close_above_hl_threshold=self.params.close_above_hl_bar_count,   # Default: 3 (bars)
    vvip_hl_min_by_point_count=self.params.vviphl_min_bypoint_count            # Default: 2 (count)
)
```
**Returns:**
- `is_hl_satisfied`: Normal HL signal (regular pivot points)
- `is_vvip_signal`: VVIP signal (Very Important Pivot HL - stronger signal)
- `break_hl_at_price`: Price at which HL was broken
- `weighted_high_m`, `weighted_high_n`, `weighted_low_m`, `weighted_low_n`: Dynamic window sizes

### Step 4: Build Scoring Parameters
```python
scoring_params = self.build_scoring_params(flattern)
```
**Determines m,n values for scoring:**
- Uses **dynamic** values if available from VipHL detection
- Falls back to **trending** defaults if `is_ma_trending == True`
- Falls back to **normal** defaults otherwise

### Step 5: Compute Scoring Metrics
```python
scoring_metrics = self.compute_scoring_metrics(scoring_params)
```
**Calculates:**
- `high_score`: Normalized score (0-1) for high pivot quality
- `low_score`: Normalized score (0-1) for low pivot quality
- `combined_score`: Weighted average of high and low scores

### Step 6: Quote Trade (Calculate Stop Loss)
```python
quoted_trade = self.quote_trade(scoring_params, scoring_metrics)
```
**Calculates stop loss:**
```python
def calculate_stop_loss_percent(self):
    stop_loss_long = min(self.data.low[0], self.data.low[-1])
    return (self.data.close[0] - stop_loss_long) / self.data.close[0] * 100
```
- Stop loss = minimum of current bar's low and previous bar's low
- Expressed as percentage from current close

### Step 7: Check Stop Loss Thresholds
```python
stoploss_below_threshold = quoted_trade.stop_loss_percent < self.close_average_percent[0] * self.p.reduce_stop_loss_threshold
vviphl_stoploss_below_threshold = quoted_trade.stop_loss_percent < self.close_average_percent[0] * self.p.vviphl_reduce_stop_loss_threshold
```

**Parameters:**
- `reduce_stop_loss_threshold`: Default = 5 (multiplier)
- `vviphl_reduce_stop_loss_threshold`: Default = 5 (multiplier)

**Meaning:**
- Stop loss must be less than 5× the average daily price movement
- This ensures we're not entering when stop loss would be too wide

---

## Final Trade Entry Decision

### Step 8: Determine Signals
```python
within_lookback_period = self.last_bar_index() - self.bar_index() <= self.p.lookback

# Normal signals
has_long_signal = is_hl_satisfied and stoploss_below_threshold

# VVIP signals
has_vvip_long_signal = is_vvip_signal and vviphl_stoploss_below_threshold
```

### Step 9: Enter Trade
```python
if within_lookback_period:
    if has_long_signal or has_vvip_long_signal:
        trade_created = self.record_trade(0, scoring_params, scoring_metrics)
        if trade_created:
            self.viphl.commit_latest_recovery_window(break_hl_at_price)
```

---

## Trade Entry Conditions Summary

### Normal Signal Entry (has_long_signal)
✅ All must be TRUE:
1. **HL Satisfied** (`is_hl_satisfied == True`)
   - Price has broken above a HL level
   - Recovery window conditions met
   - Close is above HL by threshold

2. **Stop Loss Acceptable** (`stoploss_below_threshold == True`)
   - `stop_loss_percent < close_avg_percent * 5`

3. **Within Lookback Period** (`within_lookback_period == True`)
   - Current bar is within last N bars (e.g., last 600 bars)

### VVIP Signal Entry (has_vvip_long_signal)
✅ All must be TRUE:
1. **VVIP Signal** (`is_vvip_signal == True`)
   - Multiple pivot points confirming the HL level
   - Stronger signal than normal HL
   - Min by-point count reached (default: 2)

2. **Stop Loss Acceptable** (`vviphl_stoploss_below_threshold == True`)
   - `stop_loss_percent < close_avg_percent * 5`

3. **Within Lookback Period** (`within_lookback_period == True`)

---

## Key Parameters Reference

### Entry Point Parameters (入场点设置)
| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `only_body_cross` | True | Boolean | Only count body crosses (not wicks) |
| `close_above_hl_threshold` | 0.25 | Multiplier | Close must be above HL by 0.25× CA% |
| `close_above_low_threshold` | 1.25 | Multiplier | Close must be above low by 1.25× CA% |
| `close_above_recover_low_threshold` | 1.25 | Multiplier | Close above recovery low threshold |
| `low_above_hl_threshold` | 0.5 | Multiplier | Low must be above HL by 0.5× CA% |
| `hl_extend_bar_cross_threshold` | 6 | Bars | Max bars to extend HL search |
| `close_above_hl_search_range` | 5 | Bars | How many bars to look back for close above HL |
| `close_above_hl_bar_count` | 3 | Bars | Min consecutive bars with close above HL |
| `trap_recover_window_threshold` | 6 | Bars | Bars needed to confirm recovery |
| `signal_window` | 2 | Bars | Window for signal confirmation |

### Stop Loss Parameters (Reduce stop loss)
| Parameter | Default | Multiplier | Description |
|-----------|---------|------------|-------------|
| `reduce_stop_loss_threshold` | 5 | × CA% | Normal signal: stop loss must be < 5× daily avg |
| `vviphl_reduce_stop_loss_threshold` | 5 | × CA% | VVIP signal: stop loss must be < 5× daily avg |

### VVIPHL Parameters
| Parameter | Default | Count | Description |
|-----------|---------|-------|-------------|
| `vviphl_min_bypoint_count` | 2 | # pivots | Min pivot points to qualify as VVIP |

### Close Average (CA) Parameters
| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `close_avg_percent_lookback` | 200 | Bars | Lookback period for calculating avg daily % change |
| `hl_overlap_ca_percent_multiplier` | 1.5 | Multiplier | HL overlap tolerance |

### Pivot Point Parameters (By Point设置)
| Parameter | Default (Normal) | Default (Trending) | Description |
|-----------|------------------|-------------------|-------------|
| `high_by_point_n` | 10 | 5 | Left bars for high pivot |
| `high_by_point_m` | 10 | 5 | Right bars for high pivot |
| `low_by_point_n` | 8 | 4 | Left bars for low pivot |
| `low_by_point_m` | 8 | 4 | Right bars for low pivot |
| `mn_start_point_high` | 4 | 4 | Starting window for dynamic high detection |
| `mn_cap_high` | 20 | 20 | Max window for dynamic high detection |
| `mn_start_point_low` | 4 | 4 | Starting window for dynamic low detection |
| `mn_cap_low` | 20 | 20 | Max window for dynamic low detection |

### HL Violation Parameters
| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `bar_count_to_by_point` | 300 | Bars | Max bars back to search for pivot points |
| `bar_cross_threshold` | 5 | Bars | Min bars needed to establish HL violation |
| `hl_length_threshold` | 300 | Bars | Max length of HL line before invalidation |

### Lookback Parameters
| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `lookback` | 600 | Bars | Only trade in last N bars |

---

## Calculation Details

### Close Average Percent (CA%)
```python
self.close_average_percent = CloseAveragePercent(close_avg_percent_lookback=self.p.close_avg_percent_lookback)
```
- Calculates average daily percentage change over last 200 bars (default)
- Used as baseline for thresholds
- Example: If CA% = 2%, then:
  - `close_above_hl_threshold` = 0.25 × 2% = 0.5%
  - `reduce_stop_loss_threshold` = 5 × 2% = 10%

### MA Trending Detection
```python
self.ma10 = bt.indicators.SMA(self.data.close, period=10)
self.ma40 = bt.indicators.SMA(self.data.close, period=40)
self.ma100 = bt.indicators.SMA(self.data.close, period=100)
self.trending_ma_delta = self.ma10 - self.ma100
self.trending_ma_delta_distr = PercentileNearestRank(
    self.trending_ma_delta,
    period=self.p.trending_ma_delta_distr_lookback,  # Default: 500
    percentile=self.p.trending_ma_delta_distr_threshold  # Default: 1
)

self.is_ma_greater = bt.And(
    bt.Cmp(self.ma10, self.ma40) == 1,
    bt.Cmp(self.ma40, self.ma100) == 1
)
self.is_ma_trending = bt.And(
    self.is_ma_greater,
    bt.Cmp(self.trending_ma_delta, self.trending_ma_delta_distr) == 1
)
```

**Trending = TRUE when:**
1. MA10 > MA40 > MA100 (bullish alignment)
2. MA spread (MA10 - MA100) is in top 1% of historical spreads

---

## Trade Trigger Logic Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    next() - Each Bar                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Update VipHL State                                      │
│     - Bar index, MA trending, Close Average %               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Update Recovery Window                                  │
│     - Check if price "stood firm" above HL                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Check Recovery Window v3                                │
│     - Returns: is_hl_satisfied, is_vvip_signal              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Build Scoring Params                                    │
│     - Get m,n values (dynamic/trending/normal)              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Compute Scoring Metrics                                 │
│     - Calculate high_score, low_score, combined_score       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Quote Trade                                             │
│     - Calculate stop_loss_percent                           │
│       = (close - min(low[0], low[-1])) / close * 100        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  7. Check Thresholds                                        │
│     - Normal: stop_loss < CA% × 5                           │
│     - VVIP:   stop_loss < CA% × 5                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  8. Final Decision                                          │
│                                                             │
│  within_lookback_period?                                    │
│         ├── NO  → Skip                                      │
│         └── YES → Continue                                  │
│                                                             │
│  has_long_signal = is_hl_satisfied AND stoploss_ok         │
│  has_vvip_signal = is_vvip_signal AND vvip_stoploss_ok     │
│                                                             │
│  IF (has_long_signal OR has_vvip_signal):                  │
│      → ENTER TRADE ✅                                       │
│  ELSE:                                                      │
│      → NO TRADE ❌                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Example Scenario

### Market Conditions:
- Close = 50000
- Low[0] = 49500
- Low[-1] = 49600
- CA% = 2% (average daily movement)
- MA10 > MA40 > MA100 ✅ (trending)
- Lookback period active ✅

### Calculations:
1. **Stop Loss:**
   ```
   stop_loss_long = min(49500, 49600) = 49500
   stop_loss_percent = (50000 - 49500) / 50000 × 100 = 1%
   ```

2. **Thresholds:**
   ```
   reduce_stop_loss_threshold = 2% × 5 = 10%
   Stop loss (1%) < Threshold (10%) ✅
   ```

3. **HL Satisfied:**
   - Price broke above HL level
   - Recovery window confirmed (6 bars above HL)
   - Close > HL + (0.25 × 2%) = HL + 0.5%
   - `is_hl_satisfied = True` ✅

4. **Signal:**
   ```
   has_long_signal = True AND True = True ✅
   ```

5. **Result:** **ENTER TRADE** ✅

---

## Notes

### CA% (Close Average Percent)
This is the **key normalizing factor** for all thresholds. It represents the average daily price movement:
- Small CA% (e.g., 1%) → tighter thresholds
- Large CA% (e.g., 5%) → wider thresholds

### Trending vs Normal
When `is_ma_trending = True`:
- Uses smaller pivot windows (more responsive)
- May use different defaults for m,n values
- Same stop loss thresholds apply

### VVIP vs Normal
- **Normal HL**: Single pivot point confirmation
- **VVIP HL**: Multiple pivot points (min 2) confirming same level
- Both use same stop loss threshold (5× CA%)

### Dynamic vs Static Windows
- **Dynamic**: VipHL algorithm finds optimal m,n for each pivot
- **Trending Default**: Falls back to trending parameters if dynamic fails
- **Normal Default**: Falls back to normal parameters if not trending

---

## Summary

**A trade is entered when:**

1. ✅ Price has broken and recovered above a HL level (normal or VVIP)
2. ✅ Stop loss is reasonable (< 5× average daily movement)
3. ✅ Within the lookback trading window

**The quality of the trade is scored by:**
- Pivot window sizes (m, n)
- Whether it's trending or normal
- Combined score → determines position size (PnL scale)

**The position size is determined by:**
- Combined score (0-1) → PnL scale (1-3)
- Available cash / 100
- PnL scale multiplier
