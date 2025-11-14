import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from datetime import datetime
from backtrader import num2date


def plot_trade_results(dataframe, trade_list, trade_scales, lines_info, result_stats,
                       title="Trade Visualization", save_filename=None):
    """
    Visualize trading strategy results with entry/exit markers and performance statistics

    Parameters:
    - dataframe: pandas DataFrame with OHLCV data (index: datetime)
    - trade_list: list of TradeV2 objects from strategy
    - trade_scales: dict mapping trade id to PnL scale (1-3)
    - lines_info: list of tuples (hl_value, start_bar_index, end_bar_index) for VipHL lines
    - result_stats: dict with strategy performance metrics
    - title: chart title
    - save_filename: optional filename to save the plot
    """

    # Setup figure
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot price line
    ax.plot(dataframe.index, dataframe['close'], label='Close Price',
            color='darkblue', linewidth=1.5, alpha=0.6, zorder=1)

    # Plot VipHL support/resistance lines
    if lines_info:
        for hl_value, start_idx, end_idx in lines_info:
            # Get datetime range for the HL line
            start_date = dataframe.index[min(start_idx, len(dataframe)-1)]
            end_date = dataframe.index[min(end_idx, len(dataframe)-1)]

            ax.plot([start_date, end_date], [hl_value, hl_value],
                   color='purple', linestyle='--', linewidth=1.5,
                   alpha=0.4, zorder=2)

    # Add one label for VipHL lines in legend
    if lines_info:
        ax.plot([], [], color='purple', linestyle='--', linewidth=1.5,
               alpha=0.4, label='VipHL Lines')

    # Track y-axis limits for optimal display
    price_min = dataframe['close'].min()
    price_max = dataframe['close'].max()
    price_range = price_max - price_min

    # Plot trade entry and exit markers
    for idx, trade in enumerate(trade_list):
        # Get trade info
        entry_date = num2date(trade.entry_time)
        entry_price = trade.entry_price
        scale = trade_scales.get(id(trade), 1.0)
        pnl = trade.pnl

        # Calculate marker size based on PnL scale (1-3 -> size 100-300)
        marker_size = 100 + (scale - 1.0) * 100

        # Entry marker (blue, always)
        ax.scatter(entry_date, entry_price, s=marker_size,
                  color='blue', marker='^', alpha=0.7,
                  edgecolors='darkblue', linewidths=1.5, zorder=5)

        # Annotate entry with trade number
        ax.annotate(f'#{idx+1}',
                   xy=(entry_date, entry_price),
                   xytext=(0, 15), textcoords='offset points',
                   fontsize=9, color='blue', fontweight='bold',
                   ha='center', bbox=dict(boxstyle='round,pad=0.3',
                   facecolor='lightblue', alpha=0.7, edgecolor='blue'))

        # Exit markers (if trade is closed)
        if not trade.is_open:
            # Determine exit color based on profitability
            exit_color = 'green' if pnl > 0 else 'red'
            exit_dark_color = 'darkgreen' if pnl > 0 else 'darkred'

            # First exit (partial or full)
            if trade.first_time > 0:
                first_exit_date = num2date(trade.first_time)
                first_exit_price = entry_price * (1 + trade.first_return / 100)

                # Plot exit marker
                ax.scatter(first_exit_date, first_exit_price, s=marker_size,
                          color=exit_color, marker='v', alpha=0.7,
                          edgecolors=exit_dark_color, linewidths=1.5, zorder=5)

                # Draw line from entry to exit
                ax.plot([entry_date, first_exit_date],
                       [entry_price, first_exit_price],
                       color=exit_color, linestyle=':', linewidth=1.5,
                       alpha=0.5, zorder=3)

                # Annotate PnL
                y_offset = -20 if pnl < 0 else 15
                ax.annotate(f'{pnl:.1f}%',
                           xy=(first_exit_date, first_exit_price),
                           xytext=(0, y_offset), textcoords='offset points',
                           fontsize=9, color=exit_dark_color, fontweight='bold',
                           ha='center', bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='lightgreen' if pnl > 0 else 'lightcoral',
                           alpha=0.7, edgecolor=exit_dark_color))

            # Second exit (if partial exit occurred)
            if trade.take_profit and trade.second_time > 0:
                second_exit_date = num2date(trade.second_time)
                second_exit_price = entry_price * (1 + trade.second_return / 100)

                # Plot second exit marker (slightly smaller)
                ax.scatter(second_exit_date, second_exit_price, s=marker_size*0.8,
                          color=exit_color, marker='v', alpha=0.6,
                          edgecolors=exit_dark_color, linewidths=1.5, zorder=5)

                # Draw line from first exit to second exit
                if trade.first_time > 0:
                    first_exit_date = num2date(trade.first_time)
                    first_exit_price = entry_price * (1 + trade.first_return / 100)
                    ax.plot([first_exit_date, second_exit_date],
                           [first_exit_price, second_exit_price],
                           color=exit_color, linestyle=':', linewidth=1.5,
                           alpha=0.4, zorder=3)

    # Add statistics textbox
    stats_text = f"""Trade Statistics:
━━━━━━━━━━━━━━━━━━━━
Total PnL: {result_stats.get('Total Pnl%', 0):.2f}%
Avg PnL per Trade: {result_stats.get('Avg Pnl% per entry', 0):.2f}%
Trade Count: {result_stats.get('Trade Count', 0)}
Win Rate: {result_stats.get('Winning entry%', 0):.2f}%
Avg Winner: {result_stats.get('Avg Winner%', 0):.2f}%
Avg Loser: {result_stats.get('Avg Loser%', 0):.2f}%
Fit Score: {result_stats.get('Fit Score', 0):.2f}
PnL Scale: {result_stats.get('Scale', 0):.3f}"""

    # Position textbox in top-left corner
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', horizontalalignment='left',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8,
                    edgecolor='black', linewidth=1.5))

    # Create custom legend
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', label='Entry',
              markerfacecolor='blue', markeredgecolor='darkblue',
              markersize=10, markeredgewidth=1.5),
        Line2D([0], [0], marker='v', color='w', label='Exit (Profit)',
              markerfacecolor='green', markeredgecolor='darkgreen',
              markersize=10, markeredgewidth=1.5),
        Line2D([0], [0], marker='v', color='w', label='Exit (Loss)',
              markerfacecolor='red', markeredgecolor='darkred',
              markersize=10, markeredgewidth=1.5),
        Line2D([0], [0], color='darkblue', linewidth=1.5,
              alpha=0.6, label='Close Price'),
    ]

    if lines_info:
        legend_elements.append(
            Line2D([0], [0], color='purple', linestyle='--',
                  linewidth=1.5, alpha=0.4, label='VipHL Lines')
        )

    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    # Format chart
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add some padding to y-axis for better visibility
    ax.set_ylim(price_min - price_range * 0.05, price_max + price_range * 0.1)

    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save if filename provided
    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Trade visualization saved as: {save_filename}")

    # Show plot
    plt.show()

    return fig, ax


def plot_trade_results_from_strategy(strategy_instance, title="Trade Visualization",
                                     save_filename=None):
    """
    Convenience function to plot trade results directly from a strategy instance

    Parameters:
    - strategy_instance: VipHLStrategy instance after cerebro.run()
    - title: chart title
    - save_filename: optional filename to save the plot
    """
    # Extract data from strategy
    dataframe = strategy_instance.data
    trade_list = strategy_instance.trade_list
    trade_scales = strategy_instance.trade_scales
    lines_info = strategy_instance.lines_info
    result_stats = strategy_instance.result

    # Convert backtrader data to pandas DataFrame
    dates = [strategy_instance.data.datetime.datetime(i)
            for i in range(-len(strategy_instance.data), 0)]

    df = pd.DataFrame({
        'datetime': dates,
        'open': [strategy_instance.data.open[i] for i in range(-len(strategy_instance.data), 0)],
        'high': [strategy_instance.data.high[i] for i in range(-len(strategy_instance.data), 0)],
        'low': [strategy_instance.data.low[i] for i in range(-len(strategy_instance.data), 0)],
        'close': [strategy_instance.data.close[i] for i in range(-len(strategy_instance.data), 0)],
        'volume': [strategy_instance.data.volume[i] for i in range(-len(strategy_instance.data), 0)]
    })
    df.set_index('datetime', inplace=True)

    return plot_trade_results(df, trade_list, trade_scales, lines_info,
                             result_stats, title, save_filename)


# Example usage when importing
if __name__ == '__main__':
    print("This is a utility module for trade visualization.")
    print("Import and use plot_trade_results() or plot_trade_results_from_strategy()")
    print("\nExample:")
    print("  from trade_visualization import plot_trade_results_from_strategy")
    print("  results = cerebro.run()")
    print("  strat = results[0]")
    print("  plot_trade_results_from_strategy(strat, title='BTC Trades', save_filename='btc_trades.png')")
