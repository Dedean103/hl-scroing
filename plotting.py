import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# Add project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.wave_detection import progressive_wave_detection_with_refinement
    from utils.technical_indicators import calculate_rsi
except ImportError:
    # Handle case where this file is run independently
    pass

def find_best_textbox_position(ax, price_data, waves=None, textbox_width=0.22, textbox_height=0.35):
    """
    Find the best position for parameter textbox to minimize chart obstruction
    
    Parameters:
    - ax: matplotlib axis object
    - price_data: pandas Series of price data
    - waves: list of wave data (optional)
    - textbox_width: relative width of textbox (0-1)
    - textbox_height: relative height of textbox (0-1)
    
    Returns:
    - (x, y): optimal position coordinates in axis coordinates (0-1)
    """
    
    # Define potential positions (x, y) in axis coordinates
    # Format: (x, y, description)
    candidate_positions = [
        (0.02, 0.98, "top-left"),      # Current default
        (0.98, 0.98, "top-right"),     # Top-right corner
        (0.02, 0.02, "bottom-left"),   # Bottom-left corner  
        (0.98, 0.02, "bottom-right"),  # Bottom-right corner
        (0.02, 0.50, "middle-left"),   # Middle-left
        (0.98, 0.50, "middle-right"),  # Middle-right
        (0.50, 0.98, "top-center"),    # Top-center
        (0.50, 0.02, "bottom-center"), # Bottom-center
    ]
    
    # Get data range for analysis
    price_min, price_max = price_data.min(), price_data.max()
    price_range = price_max - price_min
    
    # Calculate data density in different regions of the chart
    # Divide chart into grid and count data points in each cell
    time_points = len(price_data)
    
    best_score = float('inf')
    best_position = (0.02, 0.98)  # Default fallback
    best_description = "top-left"
    
    for x, y, description in candidate_positions:
        # Calculate obstruction score for this position
        score = calculate_obstruction_score(x, y, textbox_width, textbox_height, 
                                          price_data, waves, ax)
        
        if score < best_score:
            best_score = score
            best_position = (x, y)
            best_description = description
    
    return best_position, best_description

def calculate_obstruction_score(x, y, width, height, price_data, waves, ax):
    """
    Calculate how much a textbox at position (x,y) would obstruct the chart
    Lower score = better position
    """
    score = 0
    
    # Convert textbox coordinates to data coordinates
    xlim = ax.get_xlim() 
    ylim = ax.get_ylim()
    
    # Adjust coordinates based on anchor point
    if x > 0.5:  # Right-anchored
        text_x_start = x - width
        text_x_end = x
        ha = 'right'
    else:  # Left-anchored
        text_x_start = x
        text_x_end = x + width
        ha = 'left'
        
    if y > 0.5:  # Top-anchored
        text_y_start = y - height
        text_y_end = y
        va = 'top'
    else:  # Bottom-anchored
        text_y_start = y
        text_y_end = y + height
        va = 'bottom'
    
    # Count data points that would be obscured
    total_points = len(price_data)
    time_span = len(price_data)
    
    # Estimate data points in textbox region
    time_start_idx = int(text_x_start * time_span)
    time_end_idx = int(text_x_end * time_span)
    time_start_idx = max(0, min(time_start_idx, time_span-1))
    time_end_idx = max(0, min(time_end_idx, time_span-1))
    
    if time_end_idx > time_start_idx:
        price_subset = price_data.iloc[time_start_idx:time_end_idx]
        if len(price_subset) > 0:
            price_min, price_max = price_subset.min(), price_subset.max()
            overall_min, overall_max = price_data.min(), price_data.max()
            price_range = overall_max - overall_min
            
            # Normalize price range to 0-1
            if price_range > 0:
                norm_price_min = (price_min - overall_min) / price_range
                norm_price_max = (price_max - overall_min) / price_range
                
                # Check overlap with textbox vertical range
                overlap_y = max(0, min(text_y_end, norm_price_max) - max(text_y_start, norm_price_min))
                overlap_x = (time_end_idx - time_start_idx) / time_span
                
                # Score based on overlap area and data density
                score += overlap_x * overlap_y * len(price_subset) * 100
    
    # Add penalty for wave obstruction
    if waves:
        for wave in waves:
            wave_indices = wave['indices']
            if len(wave_indices) > 0:
                wave_start_idx = wave_indices[0]
                wave_end_idx = wave_indices[-1]
                
                # Check if wave overlaps with textbox time range
                wave_start_norm = wave_start_idx / time_span
                wave_end_norm = wave_end_idx / time_span
                
                if (wave_start_norm < text_x_end and wave_end_norm > text_x_start):
                    # Wave is in textbox time range, add penalty
                    score += 500  # High penalty for obscuring waves
    
    # Add corner preferences (corners are usually better)
    corner_bonus = 0
    if (x < 0.1 or x > 0.9) and (y < 0.1 or y > 0.9):
        corner_bonus = -50  # Bonus for corner positions
    
    return score + corner_bonus

def plot_overview_chart(close_prices, all_waves, rsi_series, all_trigger_points, rsi_drop_threshold, rsi_rise_ratio, rsi_rise_absolute, 
                        search_window_length=50, rsi_period=14, trend_threshold=0.95, price_refinement_window=5, filename='BTC.csv'):
    """
    Plot overview chart showing all detected waves with RSI subplot
    """
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    start_date_all = close_prices.index.min().date()
    end_date_all = close_prices.index.max().date()
    
    # Subplot 1: Price Chart
    ax1.plot(close_prices.index.values, close_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)
    
    # Plot waves
    plotted_strict_label = False
    plotted_relaxed_label = False
    plotted_merged_label = False
    
    for i, wave in enumerate(all_waves):
        wave_indices = wave['indices']
        wave_type = wave['type']
        wave_points_dates = close_prices.index[wave_indices]
        
        # Color coding: red=strict, green=relaxed, magenta=merged
        if wave_type == 'strict':
            color = 'red'
        elif wave_type == 'relaxed':
            color = 'green'
        elif wave_type == 'merged':
            color = 'magenta'
        else:
            color = 'gray'  # fallback
        
        # Only show label once per type
        if wave_type == 'strict' and not plotted_strict_label:
            label = 'Strict Wave'
            plotted_strict_label = True
        elif wave_type == 'relaxed' and not plotted_relaxed_label:
            label = 'Relaxed Wave'
            plotted_relaxed_label = True
        elif wave_type == 'merged' and not plotted_merged_label:
            label = 'Merged Wave'
            plotted_merged_label = True
        else:
            label = None

        ax1.plot(wave_points_dates.values, close_prices.iloc[wave_indices].values, f'{color[0]}o-', markersize=6, label=label)
        
        # Wave description without point count
        wave_description = f'{wave_type.capitalize()} Wave {i+1}'
        
        ax1.annotate(wave_description, 
                    (wave_points_dates[0], close_prices.iloc[wave_indices[0]]), 
                    xytext=(5, 10), textcoords='offset points', 
                    fontsize=10, color=color, fontweight='bold')
        
    # Extract ticker symbol from filename (e.g., 'BTC.csv' -> 'BTC')
    ticker = filename.split('.')[0].upper()
    
    ax1.legend(loc='upper right')
    title_str = f'{ticker} Price Chart: {start_date_all} to {end_date_all} (Waves Marked)\n'
    title_str += f'RSI Drop Threshold: {rsi_drop_threshold}, RSI Rise Ratio: {rsi_rise_ratio:.2f}, RSI Rise Absolute: {rsi_rise_absolute}'
    ax1.set_title(title_str, fontsize=16)
    ax1.set_ylabel('Close Price', fontsize=12)
    ax1.grid(True)
    
    # Add parameter textbox with optimal positioning
    param_text = f"""Parameters:
• Search Window Length: {search_window_length}
• RSI Period: {rsi_period}
• RSI Drop Threshold: {rsi_drop_threshold}
• RSI Rise Ratio: {rsi_rise_ratio:.3f}
• RSI Rise Absolute: {rsi_rise_absolute}
• Trend Threshold: {trend_threshold}
• Price Refinement Window: {price_refinement_window}"""
    
    # Find optimal position for textbox
    try:
        optimal_pos, position_desc = find_best_textbox_position(ax1, close_prices, all_waves)
        x_pos, y_pos = optimal_pos
        
        # Determine text alignment based on position
        if x_pos > 0.5:
            ha = 'right'
        else:
            ha = 'left'
            
        if y_pos > 0.5:
            va = 'top'
        else:
            va = 'bottom'
            
        print(f"Textbox positioned at: {position_desc} ({x_pos:.2f}, {y_pos:.2f})")
        
    except Exception as e:
        # Fallback to default position if algorithm fails
        print(f"Using default textbox position (positioning algorithm failed: {e})")
        x_pos, y_pos = 0.02, 0.98
        ha, va = 'left', 'top'
    
    ax1.text(x_pos, y_pos, param_text, transform=ax1.transAxes, fontsize=14,
             horizontalalignment=ha, verticalalignment=va, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Subplot 2: RSI Chart
    ax2.plot(rsi_series.index.values, rsi_series.values, color='purple', label='14-Day RSI', linewidth=1.5)
    ax2.axhline(70, linestyle='--', color='lightcoral', alpha=0.7)
    ax2.axhline(30, linestyle='--', color='lightgreen', alpha=0.7)
    
    # Plot trigger points (avoid duplicates)
    plotted_triggers = set()
    
    def get_first_last_triggers(wave_indices, all_trigger_points, close_prices):
        wave_start_date = close_prices.index[wave_indices[0]]
        wave_end_date = close_prices.index[wave_indices[-1]]
        
        wave_triggers = [p for p in all_trigger_points if p['date'] >= wave_start_date and p['date'] <= wave_end_date]
        
        first_trigger = wave_triggers[0] if wave_triggers else None
        last_trigger = wave_triggers[-1] if wave_triggers else None
        
        return first_trigger, last_trigger
    
    # Plot trigger points for all waves
    for wave in all_waves:
        first_trigger, last_trigger = get_first_last_triggers(wave['indices'], all_trigger_points, close_prices)
        if first_trigger and (first_trigger['date'], first_trigger['rsi_value']) not in plotted_triggers:
            marker = 'v' if first_trigger['type'] == 'drop' else '^'
            ax2.plot(first_trigger['date'], first_trigger['rsi_value'], marker, color='orange', markersize=8, label='Trigger Point')
            plotted_triggers.add((first_trigger['date'], first_trigger['rsi_value']))
        
        if last_trigger and (last_trigger['date'], last_trigger['rsi_value']) not in plotted_triggers:
            marker = 'v' if last_trigger['type'] == 'drop' else '^'
            ax2.plot(last_trigger['date'], last_trigger['rsi_value'], marker, color='orange', markersize=8, label='Trigger Point')
            plotted_triggers.add((last_trigger['date'], last_trigger['rsi_value']))
    
    handles, labels = ax2.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax2.legend(unique_labels.values(), unique_labels.keys())
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_individual_wave(file_path, close_prices, rsi_series, wave_indices, trigger_points, plot_range_days=30, wave_number=1, wave_type="", if_plot_rsi=True):
    """
    Plot individual wave with optional RSI subplot
    """
    wave_start_date = close_prices.index[wave_indices[0]] 
    wave_end_date = close_prices.index[wave_indices[-1]]
    
    start_date = wave_start_date - pd.Timedelta(days=plot_range_days)
    end_date = wave_end_date + pd.Timedelta(days=plot_range_days)
    
    plot_prices = close_prices.loc[start_date:end_date]
    
    # Conditionally create subplots
    if if_plot_rsi:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax1.plot(plot_prices.index.values, plot_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
        ax2 = None
        ax1.plot(plot_prices.index.values, plot_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)
    
    wave_points_dates = close_prices.index[wave_indices]
    wave_points_prices = close_prices.iloc[wave_indices].values
    color = 'red' if wave_type == 'strict' else 'green' if wave_type == 'relaxed' else 'magenta'
    ax1.plot(wave_points_dates.values, wave_points_prices, 'o-', color=color, label=f'{wave_type.capitalize()} Wave', markersize=6)
    
    # Point annotations
    for j in range(len(wave_indices)):
        ax1.annotate(f'P{j}', (wave_points_dates[j], wave_points_prices[j]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=10, color=color, fontweight='bold')
    
    # Plot trigger points on price chart
    for point in trigger_points:
        if point['date'] >= start_date and point['date'] <= end_date:
            trigger_price = close_prices.loc[point['date']]
            ax1.axvline(point['date'], color='gray', linestyle='--', alpha=0.5)
            marker = 'v' if point['type'] == 'drop' else '^'
            label = 'RSI Drop Trigger' if point['type'] == 'drop' else 'RSI Rise Trigger'
            ax1.plot(point['date'], trigger_price, marker, color='orange', markersize=8, label=label)
        
    ax1.set_title(f'{file_path[:4]} {wave_type.capitalize()} Wave {wave_number}: {wave_start_date.date()} to {wave_end_date.date()}', fontsize=16)
    ax1.set_ylabel('Close Price', fontsize=12)
    handles, labels = ax1.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax1.legend(unique_labels.values(), unique_labels.keys())
    ax1.grid(True)
    
    # Only plot the RSI subplot if if_plot_rsi is True
    if if_plot_rsi:
        plot_rsi = rsi_series.loc[start_date:end_date]
        ax2.plot(plot_rsi.index.values, plot_rsi.values, color='purple', label='14-Day RSI', linewidth=1.5)
        wave_points_rsi = rsi_series.loc[wave_points_dates].values
        ax2.plot(wave_points_dates.values, wave_points_rsi, 'o', color=color, markersize=6)
        
        for j in range(len(wave_indices)):
            ax2.annotate(f'P{j} ({wave_points_rsi[j]:.2f})', 
                        (wave_points_dates[j], wave_points_rsi[j]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, color=color, fontweight='bold')
        
        # Plot trigger points on RSI chart
        for point in trigger_points:
            if point['date'] >= start_date and point['date'] <= end_date:
                marker = 'v' if point['type'] == 'drop' else '^'
                ax2.plot(point['date'], point['rsi_value'], marker, color='orange', markersize=8, label='Trigger Point')
                ax2.axvline(point['date'], color='gray', linestyle='--', alpha=0.5)
                
                rsi_value_text = f"{point['rsi_value']:.2f}"
                y_offset = -10 if point['type'] == 'drop' else 10
                ax2.annotate(rsi_value_text, (point['date'], point['rsi_value']),
                             xytext=(0, y_offset), textcoords='offset points',
                             ha='center', va='center', fontsize=10, color='orange',
                             bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.3'))
        
        ax2.axhline(70, linestyle='--', color='lightcoral', alpha=0.7)
        ax2.axhline(30, linestyle='--', color='lightgreen', alpha=0.7)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('RSI', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.grid(True)
        
        handles, labels = ax2.get_legend_handles_labels()
    plt.tight_layout()
    plt.show()

def plot_date_range_chart(filename, start_date, end_date, search_window_length=50, rsi_period=14, 
                         rsi_drop_threshold=10, rsi_rise_ratio=1/3, rsi_rise_absolute=10.0, 
                         trend_threshold=0.95, price_refinement_window=10):
    """
    Plot overview chart for a specific date range with wave detection
    
    Parameters:
    - filename (str): CSV file path (e.g., 'BTC.csv', 'ETH.csv')
    - start_date (str): Start date in format 'YYYY-MM-DD'
    - end_date (str): End date in format 'YYYY-MM-DD'
    - Other parameters: Wave detection configuration parameters
    """
    print(f"=== Date Range Chart: {filename} ===")
    print(f"Date Range: {start_date} to {end_date}")
    
    # Load and prepare data
    try:
        data = pd.read_csv(filename)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)
        close_prices = data['close']
        print(f"Loaded {len(close_prices)} data points from {filename}")
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return
    
    # Convert date strings to datetime objects for filtering
    try:
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
    except Exception as e:
        print(f"Error parsing dates: {e}")
        print("Please use format 'YYYY-MM-DD' for dates")
        return
    
    # Filter data to date range
    mask = (close_prices.index >= start_datetime) & (close_prices.index <= end_datetime)
    filtered_prices = close_prices.loc[mask]
    
    if len(filtered_prices) == 0:
        print(f"No data found in date range {start_date} to {end_date}")
        print(f"Available data range: {close_prices.index.min().date()} to {close_prices.index.max().date()}")
        return
    
    print(f"Filtered to {len(filtered_prices)} data points in specified range")
    
    # Calculate RSI for the filtered data
    rsi_series = calculate_rsi(filtered_prices, period=rsi_period)
    
    # Run wave detection on filtered data
    all_waves, all_trigger_points = progressive_wave_detection_with_refinement(
        filtered_prices, search_window_length, rsi_period, rsi_drop_threshold,
        rsi_rise_ratio, rsi_rise_absolute, trend_threshold, price_refinement_window
    )
    
    # Extract ticker symbol from filename
    ticker = filename.split('.')[0].upper()
    
    # Create the plot
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Subplot 1: Price Chart
    ax1.plot(filtered_prices.index.values, filtered_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)
    
    # Plot waves within the date range
    plotted_strict_label = False
    plotted_relaxed_label = False
    plotted_merged_label = False
    
    for i, wave in enumerate(all_waves):
        wave_indices = wave['indices']
        wave_type = wave['type']
        wave_points_dates = filtered_prices.index[wave_indices]
        
        # Color coding: red=strict, green=relaxed, magenta=merged
        if wave_type == 'strict':
            color = 'red'
        elif wave_type == 'relaxed':
            color = 'green'
        elif wave_type == 'merged':
            color = 'magenta'
        else:
            color = 'gray'
        
        # Only show label once per type
        if wave_type == 'strict' and not plotted_strict_label:
            label = 'Strict Wave'
            plotted_strict_label = True
        elif wave_type == 'relaxed' and not plotted_relaxed_label:
            label = 'Relaxed Wave'
            plotted_relaxed_label = True
        elif wave_type == 'merged' and not plotted_merged_label:
            label = 'Merged Wave'
            plotted_merged_label = True
        else:
            label = None
        
        ax1.plot(wave_points_dates.values, filtered_prices.iloc[wave_indices].values, f'{color[0]}o-', markersize=6, label=label)
        
        # Determine wave point count for annotation
        point_count = len(wave_indices)
        wave_description = f'{wave_type.capitalize()} Wave {i+1}'
        
        ax1.annotate(wave_description,
                    (wave_points_dates[0], filtered_prices.iloc[wave_indices[0]]),
                    xytext=(5, 10), textcoords='offset points',
                    fontsize=10, color=color, fontweight='bold')
    
    ax1.legend(loc='upper right')
    title_str = f'{ticker} Price Chart: {start_date} to {end_date} (Waves Marked)\n'
    title_str += f'RSI Drop Threshold: {rsi_drop_threshold}, RSI Rise Ratio: {rsi_rise_ratio:.2f}, RSI Rise Absolute: {rsi_rise_absolute}'
    ax1.set_title(title_str, fontsize=16)
    ax1.set_ylabel('Close Price', fontsize=12)
    ax1.grid(True)
    
    # Add parameter textbox
    param_text = f"""Parameters:
• Search Window Length: {search_window_length}
• RSI Period: {rsi_period}
• RSI Drop Threshold: {rsi_drop_threshold}
• RSI Rise Ratio: {rsi_rise_ratio:.3f}
• RSI Rise Absolute: {rsi_rise_absolute}
• Trend Threshold: {trend_threshold}
• Price Refinement Window: {price_refinement_window}"""
    
    ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes, fontsize=14,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Subplot 2: RSI Chart
    ax2.plot(rsi_series.index.values, rsi_series.values, color='purple', label='14-Day RSI', linewidth=1.5)
    ax2.axhline(70, linestyle='--', color='lightcoral', alpha=0.7)
    ax2.axhline(30, linestyle='--', color='lightgreen', alpha=0.7)
    
    # Plot trigger points (avoid duplicates)
    plotted_triggers = set()
    
    def get_first_last_triggers(wave_indices, all_trigger_points, close_prices):
        wave_start_date = close_prices.index[wave_indices[0]]
        wave_end_date = close_prices.index[wave_indices[-1]]
        
        wave_triggers = [p for p in all_trigger_points if p['date'] >= wave_start_date and p['date'] <= wave_end_date]
        
        first_trigger = wave_triggers[0] if wave_triggers else None
        last_trigger = wave_triggers[-1] if wave_triggers else None
        
        return first_trigger, last_trigger
    
    # Plot trigger points for all waves
    for wave in all_waves:
        first_trigger, last_trigger = get_first_last_triggers(wave['indices'], all_trigger_points, filtered_prices)
        if first_trigger and (first_trigger['date'], first_trigger['rsi_value']) not in plotted_triggers:
            marker = 'v' if first_trigger['type'] == 'drop' else '^'
            ax2.plot(first_trigger['date'], first_trigger['rsi_value'], marker, color='orange', markersize=8, label='Trigger Point')
            plotted_triggers.add((first_trigger['date'], first_trigger['rsi_value']))
        
        if last_trigger and (last_trigger['date'], last_trigger['rsi_value']) not in plotted_triggers:
            marker = 'v' if last_trigger['type'] == 'drop' else '^'
            ax2.plot(last_trigger['date'], last_trigger['rsi_value'], marker, color='orange', markersize=8, label='Trigger Point')
            plotted_triggers.add((last_trigger['date'], last_trigger['rsi_value']))
    
    handles, labels = ax2.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax2.legend(unique_labels.values(), unique_labels.keys())
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    
    # Results summary
    print(f"\n=== Results for {start_date} to {end_date} ===")
    print(f"Total waves detected: {len(all_waves)}")
    for i, wave in enumerate(all_waves):
        wave_start_date = filtered_prices.index[wave['indices'][0]].date()
        wave_end_date = filtered_prices.index[wave['indices'][-1]].date()
        print(f"Wave {i+1}: {wave['type']} - {wave_start_date} to {wave_end_date}")
    
    plt.tight_layout()
    
    # Save plot instead of showing (for headless environments)
    output_filename = f"{ticker}_{start_date}_to_{end_date}_waves.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved as: {output_filename}")
    
    # Try to show if display available, otherwise just save
    try:
        plt.show()
    except:
        print("Display not available, chart saved to file only")
    
    plt.close()

def plot_price_rsi_chart(filename, start_date, end_date, rsi_period=14):
    """
    Plot price and RSI chart for a specific date range (no wave detection)
    
    Parameters:
    - filename (str): CSV file path (e.g., 'BTC.csv', 'ETH.csv')
    - start_date (str): Start date in format 'YYYY-MM-DD'
    - end_date (str): End date in format 'YYYY-MM-DD'
    - rsi_period (int): RSI calculation period (default: 14)
    """
    print(f"=== Price & RSI Chart: {filename} ===")
    print(f"Date Range: {start_date} to {end_date}")
    
    # Load and prepare data
    try:
        data = pd.read_csv(filename)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)
        close_prices = data['close']
        print(f"Loaded {len(close_prices)} data points from {filename}")
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return
    
    # Convert date strings to datetime objects for filtering
    try:
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
    except Exception as e:
        print(f"Error parsing dates: {e}")
        print("Please use format 'YYYY-MM-DD' for dates")
        return
    
    # Filter data to date range
    mask = (close_prices.index >= start_datetime) & (close_prices.index <= end_datetime)
    filtered_prices = close_prices.loc[mask]
    
    if len(filtered_prices) == 0:
        print(f"No data found in date range {start_date} to {end_date}")
        print(f"Available data range: {close_prices.index.min().date()} to {close_prices.index.max().date()}")
        return
    
    print(f"Filtered to {len(filtered_prices)} data points in specified range")
    
    # Calculate RSI for the filtered data
    rsi_series = calculate_rsi(filtered_prices, period=rsi_period)
    
    # Extract ticker symbol from filename
    ticker = filename.split('.')[0].upper()
    
    # Create the plot
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Subplot 1: Price Chart (no waves)
    ax1.plot(filtered_prices.index.values, filtered_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)
    ax1.legend(loc='upper right')
    title_str = f'{ticker} Price Chart: {start_date} to {end_date}'
    ax1.set_title(title_str, fontsize=16)
    ax1.set_ylabel('Close Price', fontsize=12)
    ax1.grid(True)
    
    # Subplot 2: RSI Chart
    ax2.plot(rsi_series.index.values, rsi_series.values, color='purple', label=f'{rsi_period}-Day RSI', linewidth=1.5)
    ax2.axhline(70, linestyle='--', color='lightcoral', alpha=0.7, label='Overbought (70)')
    ax2.axhline(30, linestyle='--', color='lightgreen', alpha=0.7, label='Oversold (30)')
    ax2.legend(loc='upper right')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    
    # Results summary
    print(f"\n=== Price Summary for {start_date} to {end_date} ===")
    print(f"Price Range: ${filtered_prices.min():.2f} - ${filtered_prices.max():.2f}")
    print(f"Start Price: ${filtered_prices.iloc[0]:.2f}")
    print(f"End Price: ${filtered_prices.iloc[-1]:.2f}")
    price_change = ((filtered_prices.iloc[-1] / filtered_prices.iloc[0]) - 1) * 100
    print(f"Total Change: {price_change:+.2f}%")
    
    plt.tight_layout()
    
    # Save plot instead of showing (for headless environments)
    output_filename = f"{ticker}_{start_date}_to_{end_date}_price_rsi.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved as: {output_filename}")
    
    # Try to show if display available, otherwise just save
    try:
        plt.show()
    except:
        print("Display not available, chart saved to file only")
    
    plt.close()