import backtrader as bt
import numpy as np

class PercentRank(bt.Indicator):
    """
    Backtrader indicator to calculate the percent rank of the current value
    relative to its history over a given period, similar to Pine Script's ta.percentrank.
    
    The percent rank indicates what percentage of previous values were less than
    or equal to the current value.
    """
    lines = ('percent_rank',)
    params = (
        ('period', 100),  # Lookback period
    )
    plotinfo = dict(plot=False)

    def next(self):
        if len(self.data) >= self.p.period:
            # Get historical data including current value
            data_array = np.array(self.data.get(size=self.p.period))
        else:
            # If we don't have enough data yet, use what we have
            data_array = np.array(self.data.get(size=len(self.data)))

        # Remove any NaN values
        cleaned_arr = data_array[~np.isnan(data_array)]
        
        if cleaned_arr.size > 0:
            # Current value is the last value in the array
            current_value = cleaned_arr[-1]
            
            # Count how many values are less than or equal to current value
            count = np.sum(cleaned_arr < current_value)
            
            # Calculate percent rank (multiply by 100 for percentage)
            # Subtract 1 from count and length to exclude current value from both
            percent_rank = ((count) / (len(cleaned_arr))) * 100
            
            self.lines.percent_rank[0] = percent_rank
        else:
            self.lines.percent_rank[0] = float('nan')
