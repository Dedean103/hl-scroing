import backtrader as bt


class AbsROC(bt.Indicator):
    lines = ('absroc',)
    params = (('period', 1),)
    plotinfo = dict(plot=False)

    def __init__(self):
        self.addminperiod(self.params.period + 1)
        self.roc_of_close = bt.indicators.RateOfChange(self.data.close, period=1) * 100

    def next(self):
        self.lines.absroc[0] = abs(self.roc_of_close[-1])


class CloseAveragePercent(bt.Indicator):
    lines = ('value',)
    params = (
        ('close_avg_percent_lookback', 500),  # Lookback period for the SMA of absolute ROC
    )
    plotinfo = dict(plot=False)

    def __init__(self):
        # Rate of Change (ROC) of the close prices. The period is set to 1 to mimic `close[1]` in Pine Script.
        self.absROC = AbsROC()
        # Simple Moving Average (SMA) of the absolute ROC of close prices
        self.lines.value = bt.indicators.SimpleMovingAverage(
            self.absROC.absroc, period=self.p.close_avg_percent_lookback
        )

    def next(self):
        # If needed, custom logic can be added here. The calculation is primarily handled in __init__.
        pass
