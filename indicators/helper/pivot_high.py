import backtrader as bt
import numpy as np

from indicators.common.common_util import last_argmax


class PivotHigh(bt.Indicator):
    # pivothigh is the location for the drawing, value is the actual value
    lines = ('pivothigh','value')
    params = (
        ('leftbars', 2),  # Number of bars to look back
        ('rightbars', 2),  # Number of bars to look ahead
        ('distance', 0.02),  # Distance above the high to plot the arrow (as a percentage of the price)
    )

    plotinfo = dict(plot=False, subplot=False, plotlinelabels=True)
    plotlines = dict(
        pivothigh=dict(marker='v', markersize=8.0, color='red', fillstyle='full', ls='')
    )

    def __init__(self):
        self.addminperiod(self.params.leftbars + self.params.rightbars + 1)

    def next(self):
        for i in range(-self.params.leftbars, 1):
            high_window = self.data.high.get(size=self.params.leftbars + self.params.rightbars + 1, ago=i)

            if len(high_window) == self.params.leftbars + self.params.rightbars + 1:
                if last_argmax(high_window) == self.params.leftbars:
                    # Mark the pivot high with its price adjusted by 'distance' above for visual clarity
                    self.lines.pivothigh[-self.params.leftbars] = self.data.high[-self.params.leftbars] * (1 + self.params.distance)
                    self.lines.value[0] = self.data.high[-self.params.leftbars]
                else:
                    self.lines.pivothigh[-self.params.leftbars] = float('nan')
                    self.lines.value[0] = float('nan')
            else:
                self.lines.pivothigh[-self.params.leftbars] = float('nan')
                self.lines.value[0] = float('nan')
