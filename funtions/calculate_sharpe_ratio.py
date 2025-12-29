#!/usr/bin/env python3
"""
Standalone Sharpe Ratio Calculator

Calculate Sharpe ratio from BTC_daily_summary_*.csv files.

Usage:
    python calculate_sharpe_ratio.py <path_to_daily_summary.csv> [--risk_free_rate 0.04]

Example:
    python calculate_sharpe_ratio.py results/8_20_4_20_1_bar800_20251229040243/BTC_daily_summary_8_20_4_20_1_bar800_20251229040243.csv
    python calculate_sharpe_ratio.py results/8_20_4_20_1_bar800_20251229040243/BTC_daily_summary_8_20_4_20_1_bar800_20251229040243.csv --risk_free_rate 0.04
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    csv_path: str,
    risk_free_rate: float = 0.0,
    lookback: Optional[int] = None,
    lookback_date: Optional[str] = None,
    verbose: bool = True
) -> dict:
    """
    Calculate Sharpe ratio from daily summary CSV

    Args:
        csv_path: Path to the BTC_daily_summary_*.csv file
        risk_free_rate: Annual risk-free rate (e.g., 0.04 for 4%)
        lookback: Number of days to look back from the end (e.g., 600 for last 600 days)
        lookback_date: Start date for lookback period (e.g., "2023-09-01")
        verbose: Whether to print detailed statistics

    Returns:
        Dictionary containing Sharpe ratio and related statistics
    """
    csv_file = Path(csv_path)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load the data
    df = pd.read_csv(csv_file)

    # Validate required columns
    required_cols = ['Date', 'Daily Change%']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {df.columns.tolist()}")

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Apply lookback filter
    original_len = len(df)
    if lookback_date:
        # Filter by date
        lookback_dt = pd.to_datetime(lookback_date)
        df = df[df['Date'] >= lookback_dt]
        if verbose:
            print(f"\nApplying lookback date filter: {lookback_date}")
            print(f"Filtered from {original_len} to {len(df)} days")
    elif lookback is not None and lookback > 0:
        # Filter by number of days from end
        df = df.tail(lookback)
        if verbose:
            print(f"\nApplying lookback filter: last {lookback} days")
            print(f"Filtered from {original_len} to {len(df)} days")

    if df.empty:
        raise ValueError("No data remaining after applying lookback filter")

    # Get daily returns (Daily Change%)
    daily_returns = df['Daily Change%'].values

    # Remove NaN values
    daily_returns = daily_returns[~np.isnan(daily_returns)]

    if len(daily_returns) == 0:
        raise ValueError("No valid daily returns found in the CSV file")

    # Calculate daily risk-free rate (from annual)
    daily_rf_rate = ((1 + risk_free_rate) ** (1/365) - 1) * 100

    # Calculate excess returns
    excess_returns = daily_returns - daily_rf_rate

    # Calculate mean and std
    mean_return = np.mean(daily_returns)
    mean_excess = np.mean(excess_returns)
    std_returns = np.std(excess_returns, ddof=1)  # Sample standard deviation

    if std_returns == 0:
        sharpe_ratio = 0.0
        annualized_sharpe = 0.0
    else:
        # Daily Sharpe ratio
        sharpe_ratio = mean_excess / std_returns

        # Annualize (crypto trades 365 days/year)
        annualized_sharpe = sharpe_ratio * np.sqrt(365)

    # Calculate additional statistics
    total_days = len(daily_returns)
    active_trading_days = np.sum(daily_returns != 0)
    positive_days = np.sum(daily_returns > 0)
    negative_days = np.sum(daily_returns < 0)

    cumulative_return = np.prod(1 + daily_returns / 100) - 1
    max_daily_gain = np.max(daily_returns) if len(daily_returns) > 0 else 0
    max_daily_loss = np.min(daily_returns) if len(daily_returns) > 0 else 0

    # Calculate Sortino Ratio (downside risk only)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) > 0:
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        if downside_std > 0:
            sortino_ratio = (mean_excess / downside_std) * np.sqrt(365)
        else:
            sortino_ratio = 0.0
    else:
        sortino_ratio = float('inf') if mean_excess > 0 else 0.0

    results = {
        'csv_file': str(csv_file),
        'annualized_sharpe_ratio': annualized_sharpe,
        'daily_sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'mean_daily_return': mean_return,
        'daily_volatility': std_returns,
        'risk_free_rate_annual': risk_free_rate,
        'risk_free_rate_daily': daily_rf_rate,
        'total_days': total_days,
        'active_trading_days': active_trading_days,
        'positive_days': positive_days,
        'negative_days': negative_days,
        'win_rate': (positive_days / total_days * 100) if total_days > 0 else 0,
        'cumulative_return_pct': cumulative_return * 100,
        'max_daily_gain': max_daily_gain,
        'max_daily_loss': max_daily_loss,
        'date_range_start': str(df['Date'].iloc[0].date()),
        'date_range_end': str(df['Date'].iloc[-1].date()),
        'lookback_days': lookback,
        'lookback_date': lookback_date,
    }

    if verbose:
        print("\n" + "="*60)
        print(f"Sharpe Ratio Analysis")
        print("="*60)
        print(f"\nFile: {csv_file.name}")
        print(f"Date Range: {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")
        if lookback or lookback_date:
            print(f"Lookback Applied: {'Last ' + str(lookback) + ' days' if lookback else 'From ' + lookback_date}")
        print("\n--- Risk-Adjusted Returns ---")
        print(f"Annualized Sharpe Ratio:  {annualized_sharpe:>10.4f}")
        print(f"Sortino Ratio:            {sortino_ratio:>10.4f}")
        print(f"Daily Sharpe Ratio:       {sharpe_ratio:>10.4f}")

        print("\n--- Return Statistics ---")
        print(f"Mean Daily Return:        {mean_return:>10.4f}%")
        print(f"Daily Volatility (Std):   {std_returns:>10.4f}%")
        print(f"Cumulative Return:        {cumulative_return * 100:>10.2f}%")
        print(f"Max Daily Gain:           {max_daily_gain:>10.4f}%")
        print(f"Max Daily Loss:           {max_daily_loss:>10.4f}%")

        print("\n--- Trading Activity ---")
        print(f"Total Days:               {total_days:>10}")
        print(f"Active Trading Days:      {active_trading_days:>10}")
        print(f"Positive Days:            {positive_days:>10}")
        print(f"Negative Days:            {negative_days:>10}")
        print(f"Win Rate:                 {positive_days / total_days * 100:>10.2f}%")

        print("\n--- Risk-Free Rate ---")
        print(f"Annual:                   {risk_free_rate * 100:>10.2f}%")
        print(f"Daily:                    {daily_rf_rate:>10.6f}%")

        print("\n--- Sharpe Ratio Interpretation ---")
        if annualized_sharpe < 1.0:
            rating = "Poor"
        elif annualized_sharpe < 2.0:
            rating = "Good"
        elif annualized_sharpe < 3.0:
            rating = "Very Good"
        else:
            rating = "Excellent"
        print(f"Rating: {rating}")
        print("="*60 + "\n")

    return results


def batch_calculate_sharpe_ratios(
    directory: str,
    risk_free_rate: float = 0.0,
    lookback: Optional[int] = None,
    lookback_date: Optional[str] = None,
    pattern: str = "*daily_summary*.csv"
) -> pd.DataFrame:
    """
    Calculate Sharpe ratios for all daily summary CSV files in a directory

    Args:
        directory: Path to directory containing CSV files
        risk_free_rate: Annual risk-free rate
        lookback: Number of days to look back from the end
        lookback_date: Start date for lookback period
        pattern: File pattern to match (default: *daily_summary*.csv)

    Returns:
        DataFrame with Sharpe ratios for all files
    """
    dir_path = Path(directory)

    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    csv_files = list(dir_path.glob(pattern))

    if not csv_files:
        print(f"No CSV files matching pattern '{pattern}' found in {directory}")
        return pd.DataFrame()

    print(f"\nFound {len(csv_files)} CSV file(s) matching pattern '{pattern}'")
    print("="*60)

    results_list = []

    for csv_file in csv_files:
        try:
            result = calculate_sharpe_ratio(
                str(csv_file),
                risk_free_rate,
                lookback=lookback,
                lookback_date=lookback_date,
                verbose=False
            )
            results_list.append({
                'File': csv_file.name,
                'Sharpe Ratio': result['annualized_sharpe_ratio'],
                'Sortino Ratio': result['sortino_ratio'],
                'Mean Daily Return %': result['mean_daily_return'],
                'Daily Volatility %': result['daily_volatility'],
                'Cumulative Return %': result['cumulative_return_pct'],
                'Total Days': result['total_days'],
                'Win Rate %': result['win_rate'],
            })
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue

    if not results_list:
        return pd.DataFrame()

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('Sharpe Ratio', ascending=False)

    print("\n" + "="*60)
    print("Batch Sharpe Ratio Analysis Summary")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60 + "\n")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Sharpe ratio from BTC_daily_summary CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single file:
    python calculate_sharpe_ratio.py results/8_20_4_20_1_bar800_20251229040243/BTC_daily_summary_8_20_4_20_1_bar800_20251229040243.csv

  With risk-free rate:
    python calculate_sharpe_ratio.py results/8_20_4_20_1_bar800_20251229040243/BTC_daily_summary_8_20_4_20_1_bar800_20251229040243.csv --risk_free_rate 0.04

  With lookback (last 600 days):
    python calculate_sharpe_ratio.py results/8_20_4_20_1_bar800_20251229040243/BTC_daily_summary_8_20_4_20_1_bar800_20251229040243.csv --lookback 600

  With lookback date:
    python calculate_sharpe_ratio.py results/8_20_4_20_1_bar800_20251229040243/BTC_daily_summary_8_20_4_20_1_bar800_20251229040243.csv --lookback_date 2023-09-01

  Batch mode (all files in directory):
    python calculate_sharpe_ratio.py results/ --batch

  Batch mode with custom pattern:
    python calculate_sharpe_ratio.py results/ --batch --pattern "*summary*.csv"
        """
    )

    parser.add_argument(
        'path',
        help='Path to CSV file or directory (for batch mode)'
    )

    parser.add_argument(
        '--risk_free_rate',
        '-r',
        type=float,
        default=0.0,
        help='Annual risk-free rate (default: 0.0, example: 0.04 for 4%%)'
    )

    parser.add_argument(
        '--lookback',
        '-l',
        type=int,
        default=None,
        help='Number of days to look back from the end (e.g., 600 for last 600 days)'
    )

    parser.add_argument(
        '--lookback_date',
        '-d',
        type=str,
        default=None,
        help='Start date for lookback period (e.g., "2023-09-01")'
    )

    parser.add_argument(
        '--batch',
        '-b',
        action='store_true',
        help='Batch mode: process all CSV files in directory'
    )

    parser.add_argument(
        '--pattern',
        '-p',
        type=str,
        default='*daily_summary*.csv',
        help='File pattern for batch mode (default: *daily_summary*.csv)'
    )

    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output CSV file path for batch results'
    )

    args = parser.parse_args()

    try:
        if args.batch:
            # Batch mode
            results_df = batch_calculate_sharpe_ratios(
                directory=args.path,
                risk_free_rate=args.risk_free_rate,
                lookback=args.lookback,
                lookback_date=args.lookback_date,
                pattern=args.pattern
            )

            if args.output and not results_df.empty:
                output_path = Path(args.output)
                results_df.to_csv(output_path, index=False)
                print(f"Results saved to: {output_path}")
        else:
            # Single file mode
            calculate_sharpe_ratio(
                csv_path=args.path,
                risk_free_rate=args.risk_free_rate,
                lookback=args.lookback,
                lookback_date=args.lookback_date,
                verbose=True
            )

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
