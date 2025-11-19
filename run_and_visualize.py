"""
Example script to run VipHL strategy and visualize trade results
"""
import pandas as pd
import backtrader as bt
from viphl_strategy_scoring import VipHLStrategy, load_data_from_csv
from trade_visualization import plot_trade_results


def run_strategy_and_plot(csv_file='BTC.csv', save_plot=True):
    """
    Run the VipHL strategy and generate trade visualization

    Parameters:
    - csv_file: path to the OHLCV data CSV
    - save_plot: whether to save the plot as an image file
    """
    # Load data
    print(f"Loading data from {csv_file}...")
    dataframe = load_data_from_csv(csv_file)

    # Setup cerebro
    cerebro = bt.Cerebro()
    pandasData = bt.feeds.PandasData(dataname=dataframe)
    cerebro.adddata(pandasData)
    cerebro.broker.set_coc(True)

    # Add strategy with configuration
    cerebro.addstrategy(
        VipHLStrategy,
        mintick=0.01,
        
        # Dynamic mn configuration
        dynamic_mn_enabled=True,    # Enable dynamic mn detection
        dynamic_mn_start=4,         # Start testing from mn=4
        dynamic_mn_step=1,          # Increment by 1
        max_mn_cap=20,              # Maximum mn value
        
        # Static fallback parameters
        high_by_point_n=14,
        high_by_point_m=14,
        low_by_point_n=14,
        low_by_point_m=14,
        high_by_point_n_on_trend=14,
        high_by_point_m_on_trend=4,
        low_by_point_n_on_trend=14,
        low_by_point_m_on_trend=14,
        
        power_scaling_factor=1.5,
        high_score_scaling_factor=0.5,
        low_score_scaling_factor=0.5,
        
        # Debug logging settings
        debug_mode=False,           # Disable console output to avoid spam
        save_debug_to_file=True,    # Save debug logs to file
        debug_log_file="run_and_visualize_debug.txt",  # Debug log filename
        
        on_trend_ratio=1,
    )

    # Run strategy
    print("Running strategy...")
    results = cerebro.run()
    strat = results[0]

    # Print results
    print("\n" + "="*50)
    print("STRATEGY RESULTS")
    print("="*50)
    for key, value in strat.result.items():
        print(f"{key:.<30} {value}")
    print("="*50 + "\n")

    # Generate visualization
    print("Generating trade visualization...")
    ticker = csv_file.split('.')[0].upper()
    title = f"{ticker} VipHL Strategy - Trade Results"

    save_filename = None
    if save_plot:
        save_filename = f"{ticker}_trade_results.png"

    # Extract strategy parameters for display
    strategy_params = {
        'dynamic_mn_enabled': strat.params.dynamic_mn_enabled,
        'dynamic_mn_start': strat.params.dynamic_mn_start,
        'max_mn_cap': strat.params.max_mn_cap,
        'high_by_point_n': strat.params.high_by_point_n,
        'high_by_point_m': strat.params.high_by_point_m,
        'low_by_point_n': strat.params.low_by_point_n,
        'low_by_point_m': strat.params.low_by_point_m,
        'high_by_point_n_on_trend': strat.params.high_by_point_n_on_trend,
        'high_by_point_m_on_trend': strat.params.high_by_point_m_on_trend,
        'low_by_point_n_on_trend': strat.params.low_by_point_n_on_trend,
        'low_by_point_m_on_trend': strat.params.low_by_point_m_on_trend,
        'enable_scoring_scale': strat.params.enable_scoring_scale,
    }

    plot_trade_results(
        dataframe=dataframe,
        trade_list=strat.trade_list,
        trade_scales=strat.trade_scales,
        lines_info=strat.lines_info,
        result_stats=strat.result,
        title=title,
        save_filename=save_filename,
        strategy_params=strategy_params
    )

    return strat, dataframe


if __name__ == '__main__':
    # Run with BTC data
    run_strategy_and_plot('BTC.csv', save_plot=True)

    # Uncomment to run with ETH data
    # run_strategy_and_plot('ETH.csv', save_plot=True)
