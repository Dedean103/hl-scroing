if __name__ == '__main__':
    df = load_data_from_csv(file)

    all_results = []
    print(f"Running {len(param_combinations)} parameter combinations...\n")

    for i, params in enumerate(param_combinations):
        #if i > 4089:

            cerebro = bt.Cerebro()
            data = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data)
            cerebro.broker.set_coc(True)
            cerebro.addstrategy(VipHLStrategy, **params)

            try:
                results = cerebro.run()
                strat = results[0]
                score = strat.result.get('Fit Score', None)
                totl_pnl = strat.result.get('Total Pnl%', None)
                avg_pnl = strat.result.get('Avg Pnl% per entry', None)
                trade_count = strat.result.get('Trade Count', None)
                

                result = {
                    **params,
                    'Fit Score': score,
                    'Total Pnl%': totl_pnl,
                    'Avg Pnl% per entry': avg_pnl,
                    'Trade Count':trade_count
                }
            except Exception as e:
                result = {
                    **params,
                    'Fit Score': None,
                    'Total Pnl%': None,
                    'Avg Pnl% per entry': None,
                    'Trade Count': None,

                    'Error': str(e)
                }

            all_results.append(result)

            # Print progress
            if result.get('Fit Score') is not None:
                print(f"{i+1:3}/{len(param_combinations)} | Fit Score: {result['Fit Score']:.4f} | Params: {params}")
            else:
                print(f"{i+1:3}/{len(param_combinations)} | Error: {result.get('Error')} | Params: {params}")

            # Save every 10 iterations
            if i % 10 == 0:
                output_path = 'grid_search_results.csv'
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                    writer.writeheader()
                    writer.writerows(all_results)