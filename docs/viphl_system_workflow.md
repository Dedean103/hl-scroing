# VipHL Trading System - Complete Workflow Documentation

## System Overview

The VipHL (Very Important Pivot High/Low) trading system is a sophisticated algorithmic trading strategy that identifies pivot points with dynamic window sizing, builds support/resistance levels, detects break-and-recover patterns, and executes trades with dynamic position sizing based on the strength of the underlying market structure.

## Key Concepts

### Understanding the Real-Time Processing Model

The system operates in a **streaming/incremental manner**, processing each bar as it arrives in real-time, rather than as a batch post-processing operation. This is a critical distinction for understanding how the system works.

**Common Misconception**: 
> "For each bar index (e.g., 650), the minimum mn would be tested [650-4:650+4]. If successful, verify again with [650-5:650+5]... This continues to the end of data (1434). Then repeat for 651, 652, etc."

**Actual Implementation**:
- The system doesn't wait until all data is loaded to process bar 650
- Instead, pivots are detected and confirmed **in real-time** as bars arrive
- Bar 650 starts being evaluated as a potential pivot when bar 654 arrives (650+4)
- Each new bar triggers evaluation of new candidates and expansion of existing ones

## Complete Workflow

### 1. Pivot Detection with Dynamic Window Expansion

#### Initial Candidate Creation
When a new bar arrives (current_bar), the system checks if the bar at position (current_bar - mn_start) could be a pivot:

```
Example: At current_bar=20, check if bar 16 is highest in [12:20] (with mn_start=4)
```

#### Progressive Expansion
If the candidate passes the initial test:
- The candidate is added to active tracking
- On each subsequent bar, the window expands by 1 on each side
- Expansion continues until:
  - The candidate is no longer the highest/lowest (failure)
  - The window reaches mn_cap (default=20)
  - Not enough bars exist to expand further

```
Bar 16 timeline:
- current_bar=20: Candidate added with mn=4
- current_bar=21: Expanded to mn=5 (still highest)
- current_bar=22: Expanded to mn=6 (still highest)
- current_bar=23: Expanded to mn=7 (still highest)
- current_bar=24: Failed - no longer highest, confirmed with mn=7
```

#### Confirmation Process
When a pivot candidate fails or reaches the cap:
1. **Rejection**: Marked as rejected with reason (e.g., "lost_high_status")
2. **Confirmation**: Immediately confirmed with its last successful (m,n) value
3. **Output**: The pivot indicator outputs the confirmed values
4. **Removal**: Candidate removed from active tracking

### 2. ByPoint Creation and Filtering

After a pivot is confirmed by the indicator, it becomes a ByPoint if it passes several filters:

#### Filtering Checks
1. **Trend Alignment**: 
   - Checks if pivot occurred during expected market conditions (trending/normal)
   - Uses MA10 > MA40 > MA100 for trend detection

2. **Recency Filter**: 
   - Only processes pivots within `bar_count_to_by_point` (default=800 bars)
   - This is why ByPoints only start appearing around bar 650 in a 1443-bar dataset

3. **Debug Ranges** (optional):
   - Price and time filters for testing specific scenarios

#### ByPoint Structure
Each ByPoint stores:
- **Pivot price and location**: price, bar_index_at_pivot, bar_time_at_pivot
- **Dynamic window sizes**: n (left bars), m (right bars) 
- **Context**: is_high/is_low, is_trending, close_avg_percent
- **Usage tracking**: used flag for HL construction

Example from debug trace:
```
ByPoint detected at current_bar=650:
- pivot_bar: 639
- type: high
- price: 27483.57
- n: 11, m: 11  (dynamically determined)
- trending: False
```

### 3. HL (Horizontal Level) Construction and Merging

HLs are support/resistance levels built from ByPoints through a sophisticated merging process.

#### Initial HL Creation
- Each new ByPoint initially creates its own HL
- HL inherits the ByPoint's price, n, m values
- Starts and ends at the pivot bar

#### Merge Evaluation
For each new HL, the system checks against existing HLs for merge potential:

1. **Same Type Check**: Both must be highs or both lows
2. **Overlap Check**: Time ranges must overlap
3. **Bar Cross Check**: Number of price crosses between them < threshold
4. **Length Check**: Combined length < threshold (default=300 bars)

#### Merge Process
When conditions are met:
```python
# Weighted averaging for price
weight_last = 3
weight_second_last = 2  
weight_others = 1
merged_value = calculate_weighted_hl_value(prices, weights)

# Similar weighting for n,m values
weighted_n = weighted_average(all_n_values, weights)
weighted_m = weighted_average(all_m_values, weights)
```

Example from trace:
```
HL merged at bar 701:
- Original HL from bar 674
- Merged with new ByPoint
- Result: weighted_value=34992.00, weighted_n=6, weighted_m=6
- Now contains 2 ByPoints
```

### 4. Trade Signal Generation

The system monitors HLs for break-and-recover patterns that trigger trades.

#### A. Recovery Window Detection
1. **Break Detection**: Price breaks below an HL
2. **Recovery Monitoring**: Tracks if price recovers above HL
3. **Window Validation**: Must recover within `trap_recover_window_threshold` (6 bars)

#### B. Signal Types

**Normal Signal (`is_hl_satisfied`):**
- Price recovers above HL + threshold
- Close > low * (1 + volatility * threshold)  
- Recovery happens within window

**VIP Signal (`is_vvip_signal`):**
- Stronger setup with multiple ByPoints
- Higher confidence configuration

### 5. Trade Execution with Dynamic Scoring

When a signal triggers, the system calculates position size and PnL scaling based on the strength of the underlying HL structure.

#### Scoring Calculation
The combined score reflects the actual (m,n) windows that built the triggering HL:

```python
# Get actual m,n from the HL
high_m, high_n = hl.weighted_high_m, hl.weighted_high_n
low_m, low_n = hl.weighted_low_m, hl.weighted_low_n

# Calculate normalized scores (0-1 range)
high_score = (high_m^k + high_n^k) / (2 * max_cap^k)
low_score = (low_m^k + low_n^k) / (2 * max_cap^k)

# Combine with scaling factors
combined_score = (high_score * high_weight + low_score * low_weight) / 2
```

#### Position Sizing
Larger (m,n) values indicate stronger pivots, leading to larger positions:

```python
base_size = floor(order_size_usd / current_price)
entry_size = floor(base_size * combined_score)
```

#### PnL Scaling
Returns are amplified for trades from stronger structures:

```python
pnl_scale = 1 + 2 * combined_score
final_pnl = percent_return * entry_size * pnl_scale
```

### 6. Trade Management

Once entered, trades are managed with:

1. **Stop Loss**: Exit if price < min(low[-1], low[-2])
2. **Profit Taking**: Partial exits at volatility multiples
3. **Max Duration**: Force exit after 6 months
4. **Tracking**: Max drawdown, equity curve, fit score calculation

## Key Implementation Files

### Core Strategy
- `hl-scroing/viphl_strategy_scoring.py`: Main strategy implementation with Backtrader
- `hl-scroing/run_viphl_and_plot.py`: Runner with visualization

### Indicators
- `viphl-source-code/indicators/helper/pivot_high.py`: Dynamic high pivot detector
- `viphl-source-code/indicators/helper/pivot_low.py`: Dynamic low pivot detector

### Data Structures
- `viphl-source-code/indicators/viphl/dto/viphl.py`: VipHL class with HL construction
- `viphl-source-code/indicators/viphl/dto/bypoint.py`: ByPoint data structure
- `viphl-source-code/indicators/viphl/dto/hl.py`: HL (Horizontal Level) structure
- `dto/trade_v2.py`: Trade tracking and management

## Configuration Parameters

### Dynamic Window Parameters
- `mn_start_point_high/low`: Starting window size (default=4)
- `mn_cap_high/low`: Maximum window size (default=20)
- `mn_start_point_high_trend/low_trend`: Trending market variants

### HL Construction
- `bar_count_to_by_point`: Recent history window (default=800)
- `bar_cross_threshold`: Max crosses for merge (default=5)
- `hl_length_threshold`: Max combined length (default=300)

### Trade Signals
- `trap_recover_window_threshold`: Recovery window (default=6 bars)
- `close_above_hl_threshold`: Entry threshold (default=0.25%)
- `signal_window`: Confirmation window (default=2 bars)

### Scoring & Sizing
- `enable_hl_byp_scoring`: Enable dynamic scoring
- `power_scaling_factor`: Exponent k for scoring (default=1.0)
- `high/low_score_scaling_factor`: Weight multipliers

## Debug and Analysis Tools

### Markdown Debug Logs
When `debug_mode=True` and `debug_log_path` is set:
- Logs every ByPoint detection with actual m,n values
- Tracks HL creation and merging events
- Records trade entries with scoring details
- Creates auditable trace of system decisions

### Visualization
The `run_viphl_and_plot.py` script provides:
- Price charts with HL levels
- Trade entry/exit markers
- Performance statistics
- Backtest results

## Summary

The VipHL system represents a sophisticated approach to algorithmic trading that:
1. **Dynamically identifies** market structure through expanding pivot detection
2. **Builds adaptive** support/resistance levels with weighted merging
3. **Detects high-probability** setups through break-and-recover patterns
4. **Scales positions** based on the strength of underlying market structure
5. **Manages risk** through systematic stop-loss and profit-taking rules

The key innovation is the dynamic (m,n) detection that adapts to market conditions, providing more accurate pivot identification than static windows, which then flows through to more intelligent position sizing and risk management.