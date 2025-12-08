# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a cryptocurrency trading strategy system built on Backtrader that implements VipHL (Very Important Pivot High/Low) pattern detection for algorithmic trading. The system identifies pivot points and trading signals based on moving average trends and price action patterns.

## Core Architecture

### Main Components

1. **VipHLStrategy** (`viphl_strategy.py`): Main trading strategy class that extends `bt.Strategy`
   - Implements VipHL pattern detection and signal generation
   - Manages trade execution, position sizing, and risk management
   - Integrates pivot point detection with moving average trend analysis

2. **Custom Indicators** (`indicators/`):
   - `PivotHigh`/`PivotLow` (`indicators/helper/`): Identifies local price extremes
   - `CloseAveragePercent` (`indicators/helper/close_average.py`): Average price percentage calculations
   - `PercentileNearestRank` (`indicators/helper/percentile_nearest_rank.py`): Statistical ranking

3. **Data Structures** (`dto/`):
   - `TradeV2` (`dto/trade_v2.py`): Comprehensive trade tracking with entry/exit management
   - Supports partial exits, stop losses, and profit taking

4. **Data Sources**:
   - **TradingView Integration** (`tvDatafeed/`): Real-time data fetching from TradingView
   - **CSV Support**: Static data loading from BTC.csv, ETH.csv files

5. **Backtrader Framework**: Extensive customized backtrader installation with modified:
   - Analyzers, brokers, feeds, filters, indicators, observers, and plotting modules
   - Enhanced with custom implementations for specific trading requirements

## Development Commands

### Running the Strategy
```bash
# Main strategy execution
python3 viphl_strategy.py

# Install dependencies
pip install -r requirements.txt

# Setup virtual environment
python -m venv venv
```

### Dependencies
Core requirements from `requirements.txt`:
- `Matplotlib` - Plotting and visualization
- `viphl` - External VipHL pattern detection library (required)
- `pandas` - Data manipulation and analysis
- `websocket-client` - TradingView data feed connectivity
- `requests` - HTTP requests for data fetching

## Key Configuration Parameters

Located in `VipHLStrategy.params`:

**Pivot Point Detection:**
- `high_by_point_n/m`: Normal market pivot detection (10/10 for highs, 8/8 for lows)
- `high_by_point_n_on_trend/m_on_trend`: Trending market pivot detection (5/5, 4/4)

**Signal Generation:**
- `close_above_low_threshold`: 1.25% threshold for entry signals
- `trap_recover_window_threshold`: 6-bar recovery window
- `signal_window`: 2-bar signal confirmation

**Risk Management:**
- `order_size_in_usd`: $2M position sizing
- `stop_loss_pt`: 1.0% stop loss
- `cycle_month`: 6-month holding period
- `max_gain_pt`: 50% maximum gain cap

## Data Format Requirements

CSV files must contain OHLCV data with datetime index:
- `datetime` (index): Timestamp
- `open`, `high`, `low`, `close`: Price data
- `volume`: Trading volume
- `openinterest`: Open interest (can be 0)

## External Dependencies

The project requires the `viphl` module which contains:
- `VipHL` class for pattern detection
- `Settings` configuration object
- `from_recovery_window_result_v2` result processing

## Strategy Logic Flow

1. **Initialization**: Setup pivot indicators, moving averages, and VipHL engine
2. **Market Analysis**: Detect trending conditions using MA10 > MA40 > MA100
3. **Pattern Detection**: Identify VipHL patterns using pivot points
4. **Signal Generation**: Generate entry signals based on recovery windows
5. **Trade Management**: Handle partial exits, stop losses, and profit taking
6. **Results**: Calculate comprehensive performance metrics including fit score

## Testing and Development

Run the strategy with different configurations by modifying parameters in the main section:
```python
cerebro.addstrategy(
    VipHLStrategy,
    mintick=0.01,
    close_above_low_threshold=0.5,  # Example parameter modification
)
```

Use TradingView data feed for live market data or CSV files for backtesting historical data.