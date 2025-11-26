import backtrader as bt


class DTPL(bt.Indicator):
    lines = ('is_dtpl',)
    params = (
        ('dtpl_condition_lookback_period', 15),
        ('dtpl_high_lookback_period', 250),
        ('dtpl_fast_ma_length', 10),
        ('dtpl_medium_ma_length', 40),
        ('dtpl_slow_ma_length', 100),
        ('dtpl_no_hl', True),
    )
    plotinfo = dict(plot=False)

    def __init__(self):
        self.addminperiod(self.p.dtpl_slow_ma_length)
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.dtpl_fast_ma_length)
        self.medium_ma = bt.indicators.SMA(self.data.close, period=self.p.dtpl_medium_ma_length)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.dtpl_slow_ma_length)

        self.highest_dtpl_bar_offset_in_session = bt.ind.FindLastIndexHighest(self.data.close, period=self.p.dtpl_condition_lookback_period + 1)
        self.highest_dtpl_bar_offset_required = bt.ind.FindLastIndexHighest(self.data.close, period=self.p.dtpl_high_lookback_period + 1)

        self.in_dtpl_zone = bt.Cmp(self.highest_dtpl_bar_offset_in_session, self.highest_dtpl_bar_offset_required) == 0
        self.is_ma_dtpl = bt.And(
            bt.Cmp(self.fast_ma, self.medium_ma) == 1,
            bt.Cmp(self.medium_ma, self.slow_ma) == 1
        )
        self.l.is_dtpl = bt.And(self.in_dtpl_zone, self.is_ma_dtpl)
