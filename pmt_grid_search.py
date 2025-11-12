import itertools
import csv
import backtrader as bt
from viphl_strategy_scoring import VipHLStrategy, load_data_from_csv

def generate_parameter_combinations():
    """Generate all combinations of m and n parameters ranging from 4 to 10"""
    
    # Parameter ranges
    mn_range = [4, 14]  # 4 to 10 inclusive
    k_range = [1.1, 1.3, 1.5, 2]

    c = 0
    # Generate all combinations for normal and trending conditions
    param_combinations = []

    for high_n, high_m, low_n, low_m, high_n_trend, high_m_trend, low_n_trend, low_m_trend, k in itertools.product(
        mn_range, mn_range, mn_range, mn_range, mn_range, mn_range, mn_range, mn_range, k_range
    ):
    #for k in k_range:
        params = {
            'high_by_point_n': high_n,
            'high_by_point_m': high_m,
            'low_by_point_n': low_n,
            'low_by_point_m': low_m,
            'high_by_point_n_on_trend': high_n_trend,
            'high_by_point_m_on_trend': high_m_trend,
            'low_by_point_n_on_trend': low_n_trend,
            'low_by_point_m_on_trend': low_m_trend,
            'power_scaling_factor': k,
            'mintick': 0.01,
            'debug_mode': False
        }
        param_combinations.append(params)
    
    return param_combinations

def save_results_to_csv(results, filename='grid_search_results.csv'):
    """Save results to CSV file"""
    if not results:
        return
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {filename}")

def run_grid_search(csv_file='BTC.csv', save_interval=10):
    """Run grid search on VipHL strategy parameters"""
    
    # Load data
    df = load_data_from_csv(csv_file)
    
    # Generate parameter combinations
    param_combinations = generate_parameter_combinations()
    
    all_results = []
    print(f"Running {len(param_combinations)} parameter combinations...\n")
    print(f"Total combinations: {len(param_combinations)}")
    print("Starting grid search...\n")

    for i, params in enumerate(param_combinations):
        #if i > 690:
            try:
                # Create Cerebro instance
                cerebro = bt.Cerebro()
                data = bt.feeds.PandasData(dataname=df)
                cerebro.adddata(data)
                cerebro.broker.set_coc(True)
                cerebro.addstrategy(VipHLStrategy, **params)

                # Run strategy
                results = cerebro.run()
                strat = results[0]
                
                # Extract results
                result = {
                    **params,
                    'Fit Score': strat.result.get('Fit Score', None),
                    'Total Pnl%': strat.result.get('Total Pnl%', None),
                    'Avg Pnl% per entry': strat.result.get('Avg Pnl% per entry', None),
                    'Trade Count': strat.result.get('Trade Count', None),
                    'Winning entry%': strat.result.get('Winning entry%', None),
                    'Avg Winner%': strat.result.get('Avg Winner%', None),
                    'Avg Loser%': strat.result.get('Avg Loser%', None),
                    'Scale': strat.result.get('Scale', None)
                }
                
            except Exception as e:
                result = {
                    **params,
                    'Fit Score': None,
                    'Total Pnl%': None,
                    'Avg Pnl% per entry': None,
                    'Trade Count': None,
                    'Winning entry%': None,
                    'Avg Winner%': None,
                    'Avg Loser%': None,
                    'Scale': None,
                    'Error': str(e)
                }

            all_results.append(result)

            # Print progress
            if result.get('Fit Score') is not None:
                print(f"{i+1:5}/{len(param_combinations)} | Fit Score: {result['Fit Score']:.4f} | " +
                    f"Total PnL%: {result['Total Pnl%']:.2f} | Trades: {result['Trade Count']} | " +
                    f"Scale: {result['Scale']:.3f} | " +
                    f"k: {params['power_scaling_factor']:.1f} | " +
                    f"High n/m: {params['high_by_point_n']}/{params['high_by_point_m']} | " +
                    f"Low n/m: {params['low_by_point_n']}/{params['low_by_point_m']}")
            else:
                print(f"{i+1:5}/{len(param_combinations)} | Error: {result.get('Error', 'Unknown')} | " +
                    f"k: {params['power_scaling_factor']:.1f} | " +
                    f"High n/m: {params['high_by_point_n']}/{params['high_by_point_m']} | " +
                    f"Low n/m: {params['low_by_point_n']}/{params['low_by_point_m']}")

            # Save results every save_interval iterations
            if (i + 1) % save_interval == 0:
                #save_results_to_csv(all_results, f'grid_search_results_partial_{i+1}.csv')
                save_results_to_csv(all_results, 'grid_search_results_scroing.csv')
                print(f"Partial results saved at iteration {i+1}")

    
    # Print summary
    valid_results = [r for r in all_results if r.get('Fit Score') is not None]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['Fit Score'])
        print(f"\nGrid search completed!")
        print(f"Total combinations tested: {len(param_combinations)}")
        print(f"Successful runs: {len(valid_results)}")
        print(f"Failed runs: {len(param_combinations) - len(valid_results)}")
        print(f"\nBest result:")
        print(f"Fit Score: {best_result['Fit Score']:.4f}")
        print(f"Total PnL%: {best_result['Total Pnl%']:.2f}")
        print(f"Parameters: {best_result}")
    
    return all_results

if __name__ == '__main__':
    results = run_grid_search(csv_file='BTC.csv', save_interval=10)