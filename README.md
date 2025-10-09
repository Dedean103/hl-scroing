# Getting Started

To get started, run the following commands:

1.  `python -m venv venv`
2.  `pip install -r requirements.txt`
3.  `python viphl_strategy.py` 

```base
python viphl_strategy.py

{'Total Pnl%': 82.19, 'Avg Pnl% per entry': 2.28, 'Trade Count': 36, 'Winning entry%': 55.56, 'Avg Winner%': 8.3, 'Avg Loser%': -5.23, 'Fit Score': 1.98}
```

# Change Configurations

To change the strategy configurations, you can modify the parameters passed to `VipHLStrategy` in the `viphl_strategy.py` file.

For example, to change the `close_above_low_threshold` parameter, uncomment and modify the following line:

```python
    # Add the trading strategy and set config
    cerebro.addstrategy(
        VipHLStrategy,
        mintick = 0.01,
        # uncomment to change params
        close_above_low_threshold=0.5, # Modify this value
    )
```
