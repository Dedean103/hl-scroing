"""
Test script to demonstrate PnL scaling functionality
"""
import pandas as pd
import backtrader as bt
from viphl_strategy_scoring import VipHLStrategy

def load_data_from_csv(csv_file):
    dataframe = pd.read_csv(csv_file,
                                skiprows=0,
                                header=0,
                                parse_dates=True,
                                index_col=0)
    return dataframe

if __name__ == '__main__':
    dataframe = load_data_from_csv('BTC.csv')
    cerebro = bt.Cerebro()

    pandasData = bt.feeds.PandasData(dataname=dataframe)
    cerebro.adddata(pandasData)
    cerebro.broker.set_coc(True)

    # Run with debug mode to see scale calculations
    cerebro.addstrategy(
        VipHLStrategy,
        mintick = 0.01,
        max_mn_cap = 12,
        high_by_point_n= 10,
        high_by_point_m= 10,
        low_by_point_n= 10,
        low_by_point_m= 10,
        high_by_point_n_on_trend= 10,
        high_by_point_m_on_trend= 10,
        low_by_point_n_on_trend= 10,
        low_by_point_m_on_trend= 10,
        power_scaling_factor= 1.0,
        high_score_scaling_factor= 0.5,
        low_score_scaling_factor= 0.5,
        debug_mode=True,  # Enable debug to see scale calculations
        on_trend_ratio=1,
    )

    results = cerebro.run()
    strat = results[0]

    print("\n" + "="*60)
    print("BACKTEST RESULTS WITH PNL SCALING")
    print("="*60)
    print(strat.result)
    print("\nScale mapping (exponent=2):")
    print("  Score=0.0 → Scale=1.0 (min)")
    print("  Score=0.5 → Scale=1.5")
    print("  Score=1.0 → Scale=3.0 (max)")
