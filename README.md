# VipHL Trading Strategy

A cryptocurrency trading strategy system built on Backtrader that implements VipHL (Very Important Pivot High/Low) pattern detection for algorithmic trading. The system identifies pivot points and generates mean reversion signals based on support/resistance level violations and recoveries.

## 🚀 Quick Start

### Prerequisites
- Python 3.x
- Virtual environment (recommended)

### Installation & Execution
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the strategy
python3 viphl_strategy.py
```

**Expected Output:**
```
{'Total Pnl%': 82.19, 'Avg Pnl% per entry': 2.28, 'Trade Count': 36, 'Winning entry%': 55.56, 'Avg Winner%': 8.3, 'Avg Loser%': -5.23, 'Fit Score': 1.98}
```

## 📊 Strategy Overview

### Core Concept
The VipHL strategy is a **bidirectional mean reversion system** that:
- Detects support/resistance levels using pivot point analysis
- Identifies brief level violations (breakouts)
- Enters positions when price recovers back to respect the level
- Uses dynamic position sizing based on pattern confidence

### Signal Types
- **Long Signals**: Support breaks followed by recovery
- **Short Signals**: Resistance breaks followed by recovery  
- **VVIP Signals**: High-confidence patterns with multiple pivot confirmations

## 🎯 Key Features

### 1. Adaptive Pivot Detection
- **Normal Market**: High(10,10), Low(8,8) - More reliable, slower detection
- **Trending Market**: High(5,5), Low(4,4) - Faster detection for momentum conditions
- **Market Condition**: Determined by MA10 > MA40 > MA100 relationship

### 2. Dynamic Position Sizing
Uses advanced HL byP (byPoint) scoring system that adjusts position size based on:
- **Pivot Reliability**: Larger m/n values = higher confidence = larger positions
- **Market Conditions**: Trending markets get 1.5x scoring bonus
- **User Preferences**: Configurable scaling factors for high/low pivot emphasis

**Scoring Formula:**
```python
window_score = (m + n) / (2 * max_mn_cap)
combined_score = (high_score + low_score) / total_weight
entry_size = base_size * combined_score
```

### 3. Comprehensive Risk Management
- **Dynamic Stop Loss**: Based on recent price lows
- **Position Sizing**: $2M base size adjusted by confidence score
- **Profit Taking**: 33% at first target, 67% at cycle end
- **Max Holding**: 6 months maximum
- **Risk Limits**: 1% stop loss, 50% max gain

## 🔧 Configuration

### Key Parameters

**Pivot Detection:**
```python
# Normal market conditions
high_by_point_n/m = 10/10    # Resistance detection window
low_by_point_n/m = 8/8       # Support detection window

# Trending market conditions  
high_by_point_n_on_trend/m_on_trend = 5/5   # Faster resistance
low_by_point_n_on_trend/m_on_trend = 4/4    # Faster support
```

**Scoring System:**
```python
max_mn_cap = 12                    # Scoring normalization cap
high_score_scaling_factor = 0.5    # Resistance weight
low_score_scaling_factor = 0.5     # Support weight (equal weighting)
on_trend_ratio = 1.5               # Trending market bonus
debug_mode = False                 # Enable/disable debug output
```

**Signal Generation:**
```python
close_above_low_threshold = 1.25          # Recovery confirmation %
trap_recover_window_threshold = 6         # Recovery timeframe (bars)
signal_window = 2                         # Signal confirmation
```

**Trade Management:**
```python
order_size_in_usd = 2000000              # Base position size ($2M)
cycle_month = 6.0                        # Max holding period (months)
stop_loss_pt = 1.0                       # Stop loss threshold (%)
max_gain_pt = 50.0                       # Maximum gain cap (%)
```

### Modifying Parameters
To customize the strategy, edit parameters in `viphl_strategy.py`:

```python
cerebro.addstrategy(
    VipHLStrategy,
    mintick=0.01,
    max_mn_cap=12,                    # Scoring cap for m+n normalization
    high_score_scaling_factor=0.5,    # Weight for resistance patterns
    low_score_scaling_factor=0.5,     # Weight for support patterns
    debug_mode=False,                 # Enable debug output
    close_above_low_threshold=1.25,   # Recovery confirmation threshold
    on_trend_ratio=1.5,               # Trending market bonus
)
```

## 🛠️ Development Tools

### Testing & Analysis Files
- **`viphl_strategy_scoring.py`**: Alternative strategy implementation with scoring focus
- **`pmt_grid_search.py`**: Grid search optimization for parameter tuning

### Documentation
- **`scoring_system_guide.txt`**: Complete guide to the HL byP scoring system
- **`viphl_summary.txt`**: Comprehensive strategy overview and algorithm explanation
- **`implementation_summary.txt`**: Development history and implementation details
- **`CLAUDE.md`**: Project-specific instructions for Claude Code assistant

### Usage Examples
```bash
# Run grid search optimization
python3 pmt_grid_search.py

# Alternative scoring-focused strategy
python3 viphl_strategy_scoring.py

# Main strategy execution
python3 viphl_strategy.py
```

## 📁 Project Structure

```
├── viphl_strategy.py              # Main strategy implementation (CURRENT)
├── viphl_strategy_scoring.py      # Alternative scoring-focused strategy
├── pmt_grid_search.py             # Parameter optimization grid search
├── BTC.csv                        # Bitcoin price data (main dataset)
├── CLAUDE.md                      # Claude Code assistant instructions
├── scoring_system_guide.txt       # HL byP scoring system documentation
├── viphl_summary.txt              # Strategy overview and algorithm details
├── implementation_summary.txt     # Development history and progress
├── indicators/                    # Custom indicators
│   ├── common/
│   │   ├── base_indicator.py      # Base indicator class
│   │   └── common_util.py         # Shared utilities
│   └── helper/
│       ├── pivot_high.py          # Pivot high detection
│       ├── pivot_low.py           # Pivot low detection
│       ├── close_average.py       # Average price calculations
│       └── percentile_nearest_rank.py # Statistical ranking
├── dto/                           # Data structures
│   └── trade_v2.py                # Enhanced trade tracking
├── data/                          # Market data files
│   ├── BTC.csv                    # Bitcoin data (duplicate)
│   ├── ETH.csv                    # Ethereum data
│   ├── SOL.csv                    # Solana data
│   ├── grid_search_results_BTC.csv # BTC optimization results
│   ├── grid_search_results_ETH.csv # ETH optimization results
│   └── grid_search_results_SOL.csv # SOL optimization results
├── backtrader/                    # Customized backtrader framework
│   ├── analyzers/                 # Performance analysis tools
│   ├── brokers/                   # Broker implementations
│   ├── feeds/                     # Data feed handlers
│   ├── indicators/                # Technical indicators library
│   ├── observers/                 # Trade monitoring
│   ├── plot/                      # Visualization tools
│   ├── sizers/                    # Position sizing methods
│   ├── stores/                    # Data storage interfaces
│   └── strategies/                # Strategy templates
├── venv/                          # Virtual environment
└── requirements.txt               # Dependencies
```

## 📈 Recent Improvements

### ✅ HL byP Scoring System (Latest)
- **Enhanced Position Sizing**: Dynamic sizing based on pivot pattern confidence
- **Equal Weighting**: Balanced high/low pivot contribution (0.5/0.5)
- **Optimized Caps**: max_mn_cap=12 for better position scaling (min 1/3, max 83%)
- **Debug Controls**: Configurable debug output for development

### ✅ Scoring Logic Fix
- **Issue**: Parameter changes weren't affecting results
- **Solution**: Implemented condition-specific normalization
- **Result**: 50% improvement in parameter sensitivity (0.30 → 0.45 combined score)

### ✅ Market Condition Adaptation
- **Trending Detection**: MA10 > MA40 > MA100 with momentum threshold
- **Adaptive Parameters**: Different pivot windows for trending vs normal markets
- **Scoring Bonus**: 1.5x multiplier for trending market patterns

## 📊 Performance Metrics

The strategy tracks comprehensive performance metrics:
- **Total PnL%**: Overall strategy returns
- **Trade Count**: Number of executed trades
- **Win Rate**: Percentage of profitable trades
- **Fit Score**: Risk-adjusted performance metric (win_pnl/loss_pnl)
- **Average Win/Loss**: Mean returns for winning/losing trades

## 🔍 Understanding the Algorithm

### Signal Generation Process
1. **Market Analysis**: Detect trending vs normal conditions using moving averages
2. **Pivot Detection**: Identify local highs/lows using appropriate window sizes
3. **Pattern Formation**: Group related pivots into HL (support/resistance) zones
4. **Violation Detection**: Monitor for brief breaks of key levels
5. **Recovery Confirmation**: Validate price return to respect the level
6. **Risk Assessment**: Confirm acceptable stop loss distance
7. **Position Sizing**: Calculate entry size based on pattern confidence
8. **Trade Execution**: Enter position with dynamic sizing and risk management

### Why Both High and Low Pivots Matter
Even for long-only strategies, resistance (high pivots) are essential for:
- **Market Structure**: Understanding overhead resistance for realistic targets
- **Pattern Strength**: Support levels strengthened by nearby resistance confirmation
- **Risk Management**: Avoiding trades in compressed ranges
- **VVIP Signals**: High-confidence patterns require multiple pivot confirmations

## 📋 Dependencies

Core requirements from `requirements.txt`:
- **Matplotlib**: Plotting and visualization
- **viphl**: External VipHL pattern detection library (required)
- **pandas**: Data manipulation and analysis
- **websocket-client**: TradingView data feed connectivity
- **requests**: HTTP requests for data fetching

## 🚀 Next Steps

1. **Backtest Analysis**: Run strategy on different timeframes and assets
2. **Parameter Optimization**: Use testing tools to find optimal configurations
3. **Live Trading**: Integrate with TradingView data feed for real-time execution
4. **Risk Monitoring**: Implement additional risk controls and monitoring

---

For detailed algorithm explanation, see `viphl_summary.txt`  
For scoring system details, see `scoring_system_guide.txt`  
For implementation history, see `implementation_summary.txt`
