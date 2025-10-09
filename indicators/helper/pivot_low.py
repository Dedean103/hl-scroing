import backtrader as bt
import numpy as np

from indicators.common.common_util import last_argmin


class PivotLow(bt.Indicator):
    lines = ('pivotlow', 'value')
    params = (
        ('leftbars', 2),
        ('rightbars', 2),
        ('distance', 0.02),  # Distance below the low to plot the arrow (as a percentage of the price)
    )

    plotinfo = dict(plot=False, subplot=False, plotlinelabels=True)
    plotlines = dict(
        pivotlow=dict(marker='^', markersize=8.0, color='lime', fillstyle='full', ls='')
    )

    def __init__(self):
        self.addminperiod(self.params.leftbars + self.params.rightbars + 1)

    def next(self):
        for i in range(-self.params.leftbars, 1):
            low_window = self.data.low.get(size=self.params.leftbars + self.params.rightbars + 1, ago=i)

            if len(low_window) == self.params.leftbars + self.params.rightbars + 1:
                if last_argmin(low_window) == self.params.leftbars:
                    # Mark the pivot low with its price adjusted by 'distance' below for visual clarity
                    self.lines.pivotlow[-self.params.leftbars] = self.data.low[-self.params.leftbars] * (1 - self.params.distance)
                    self.lines.value[0] = self.data.low[-self.params.leftbars]
                else:
                    self.lines.pivotlow[-self.params.leftbars] = float('nan')
                    self.lines.value[0] = float('nan')
            else:
                self.lines.pivotlow[-self.params.leftbars] = float('nan')
                self.lines.value[0] = float('nan')
