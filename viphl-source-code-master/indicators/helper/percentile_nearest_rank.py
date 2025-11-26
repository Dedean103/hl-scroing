import backtrader as bt
import numpy as np

class PercentileNearestRank(bt.Indicator):
    """
    Backtrader indicator to calculate the value at or nearest to the specified
    percentile of a dataset over a given period, similar to Pine Script's ta.percentile_nearest_rank.
    """
    lines = ('percentile_nearest_rank',)
    params = (
        ('period', 500),  # Lookback period
        ('percentile', 50),  # Target percentile
    )
    plotinfo = dict(plot=False)
    def __init__(self):
        # To keep track of the rolling window of data
        # self.addminperiod(self.p.period)
        pass

    def next(self):
        if len(self.data) >= self.p.period:
            data_array = np.array(self.data.get(size=self.p.period))
        else:
            data_array = np.array(self.data.get(size=len(self.data)))

        cleaned_arr = data_array[~np.isnan(data_array)]
        if cleaned_arr.size > 0:  # Ensure array is not empty
            percentile_value = np.percentile(cleaned_arr, self.p.percentile, method="nearest")
            self.lines.percentile_nearest_rank[0] = percentile_value
        else:
            self.lines.percentile_nearest_rank[0] = float('nan')  # Handle empty case

    # alternative implementation
    # def next(self):
    #     # print(f'Current data period: {len(self.data)}')  # Check the number of data points available
    #     if len(self.data) >= self.p.period:
    #         data_window = np.array(self.data.get(size=self.p.period))
    #         # print(f'Data window: {data_window}')  # Print the data window
    #         if len(data_window) > 0:
    #             k = int(np.ceil((self.p.percentile / 100.0) * len(data_window)))
    #             k = max(min(k, len(data_window)), 1)
    #             self.lines.percentile_nearest_rank[0] = np.partition(data_window, k-1)[k-1]
    #         else:
    #             self.lines.percentile_nearest_rank[0] = float('nan')
    #     else:
    #         self.lines.percentile_nearest_rank[0] = float('nan')

