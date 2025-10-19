import math
import pandas as pd
import backtrader as bt

from backtrader import num2date
from indicators.helper.pivot_high import PivotHigh
from indicators.helper.pivot_low import PivotLow
from viphl.dto.recovery_window import from_recovery_window_result_v2
from viphl.dto.settings import Settings
from viphl.dto.viphl import VipHL
from dto.trade_v2 import TradeV2
from indicators.helper.close_average import CloseAveragePercent
from indicators.helper.percentile_nearest_rank import PercentileNearestRank


FIT_SCORE_MAX = 500 


"""
explanations: setting up parameters
"""
class VipHLStrategy(bt.Strategy):
    # list of parameters which are configurable for the strategy

    params = (
        ('mintick', 0.01),
        ('lookback', 600),
        # By Point设置
        ('high_by_point_n', 10), # n is the # of bar on the left, m is right
        ('high_by_point_m', 10),
        ('low_by_point_n', 8),
        ('low_by_point_m', 8),
        ('high_by_point_n_on_trend', 5),
        ('high_by_point_m_on_trend', 5),
        ('low_by_point_n_on_trend', 4),
        ('low_by_point_m_on_trend', 4),
        ('show_vip_by_point', True),
        ('show_closest_vip_hl', True),
        ('show_ma_trending', False),
        ('last_by_point_weight', 3),
        ('second_last_by_point_weight', 2),
        ('by_point_weight', 1),

        # HL byP Scoring设置
        ('max_mn_cap', 20),
        ('enable_hl_byp_scoring', True),
        ('on_trend_ratio', 1.5),

        # HL Violation设置
        ('bar_count_to_by_point', 800),
        ('bar_cross_threshold', 5),
        ('hl_length_threshold', 300),

        # 入场点设置
        ('only_body_cross', True),
        ('close_above_hl_threshold', 0.25),
        ('close_above_low_threshold', 1.25),
        ('close_above_recover_low_threshold', 1.25),
        ('low_above_hl_threshold', 0.5),
        ('hl_extend_bar_cross_threshold', 6),
        ('close_above_hl_search_range', 5),
        ('close_above_hl_bar_count', 3),
        ('trap_recover_window_threshold', 6),
        ('signal_window', 2),

        # Reduce stop loss
        ('reduce_stop_loss_threshold',5),
        ('vviphl_reduce_stop_loss_threshold', 5),

        # VVIPHL
        ('vviphl_min_bypoint_count', 2),

        # CA设置
        ('close_avg_percent_lookback', 200),
        ('hl_overlap_ca_percent_multiplier', 1.5),

        # By Point设置 (Trending MA Delta Distr Config)
        ('trending_ma_delta_distr_lookback', 500),
        ('trending_ma_delta_distr_threshold', 1),

        # others
        ('use_date_range', False),

        # ---------trade inputs-------
        ('order_size_in_usd', 2000000),  # Equivalent to orderSizeInUSD
        ('cycle_month', 6.0),  # Equivalent to cycleMonth
        ('stop_loss_pt', 1.0),  # Equivalent to stopLossPt
        ('first_gain_ca_multiplier', 2.0),  # Equivalent to firstGainCAMultiplier # what is this ,leverage?
        ('max_gain_pt', 50.0),  # Equivalent to maxGainPt
        ('max_exit_ca_multiplier', 3.0),  # Equivalent to maxExitCAMultiplier
        ('stop_gain_pt', 30.0),  # Equivalent to stopGainPt
        ('toggle_pnl', True), #what is this?
    )

    def log(self, txt, dt=None, doprint=True):
        ''' Logging function fot this strategy'''
        if doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def calculate_hl_byp_score(self, m, n, pivot_type='high', is_trending=False):
        '''Calculate normalized HL byP score (0-1) based on m,n parameters'''
        if not self.p.enable_hl_byp_scoring:
            return 1.0
        
        # Normalize window size to 0-1 range
        window_score = min((m + n) / (2 * self.p.max_mn_cap), 1.0)
        
        # Apply weight multipliers and condition-specific normalization (Option A)
        if is_trending:
            # Trending conditions are more significant with on_trend_ratio multiplier
            weight_multiplier = self.p.by_point_weight * self.p.on_trend_ratio
            max_possible_weight = self.p.by_point_weight * self.p.on_trend_ratio  # Trending-specific max
        else:
            # Normal conditions use base weight
            weight_multiplier = self.p.by_point_weight
            max_possible_weight = self.p.by_point_weight  # Normal-specific max
        
        # Final score incorporating weights (normalized to each condition's max)
        final_score = min(window_score * weight_multiplier / max_possible_weight, 1.0)
        
        # Debug logging
        if hasattr(self, 'data') and len(self.data) > 0:
            print(f"[DEBUG] HL Score - Type: {pivot_type}, Trending: {is_trending}, "
                  f"m+n: {m+n}, Window: {window_score:.3f}, Weight: {weight_multiplier:.3f}, "
                  f"MaxWeight: {max_possible_weight:.3f}, Final: {final_score:.3f}")
        
        return final_score

    def __init__(self):
        '''init is called once at the last row'''

        '''
        viphl
        '''
        #pivot point meaning the local max/min within the window
        # shouldnt trending be greater than normal?
        self.normal_high_by_point = PivotHigh(leftbars=self.p.high_by_point_n, rightbars=self.p.high_by_point_m)
        self.normal_low_by_point = PivotLow(leftbars=self.p.low_by_point_n, rightbars=self.p.low_by_point_m)
        self.trending_high_by_point = PivotHigh(leftbars=self.p.high_by_point_n_on_trend, rightbars=self.p.high_by_point_m_on_trend)
        self.trending_low_by_point = PivotLow(leftbars=self.p.low_by_point_n_on_trend, rightbars=self.p.low_by_point_m_on_trend)


        self.ma10 = bt.indicators.SMA(self.data.close, period=10)
        self.ma40 = bt.indicators.SMA(self.data.close, period=40)
        self.ma100 = bt.indicators.SMA(self.data.close, period=100)
        # The larger the difference, the stronger the upward momentum.
        self.trending_ma_delta = self.ma10 - self.ma100
        self.trending_ma_delta_distr = PercentileNearestRank(self.trending_ma_delta, 
                                                             period=self.p.trending_ma_delta_distr_lookback, # # of bar look back

                                                             # Computes the value of trending_ma_delta at the Xth percentile (e.g., 90th percentile if threshold = 90)
                                                             percentile=self.p.trending_ma_delta_distr_threshold) 

        # checking both ma10 > ma40 and ma40 > 100
        self.is_ma_greater = bt.And(bt.Cmp(self.ma10, self.ma40) == 1, bt.Cmp(self.ma40, self.ma100) == 1)
        # ensure ma bullish AND The MA spread is large enough to be in the top X% of historical spreads (based on percentile)
        self.is_ma_trending = bt.And(self.is_ma_greater, bt.Cmp(self.trending_ma_delta, self.trending_ma_delta_distr) == 1)

        viphl_settings = Settings(
            high_by_point_n=self.p.high_by_point_n,
            high_by_point_m=self.p.high_by_point_m,
            low_by_point_n=self.p.low_by_point_n,
            low_by_point_m=self.p.low_by_point_m,
            high_by_point_n_on_trend=self.p.high_by_point_n_on_trend,
            high_by_point_m_on_trend=self.p.high_by_point_m_on_trend,
            low_by_point_n_on_trend=self.p.low_by_point_n_on_trend,
            low_by_point_m_on_trend=self.p.low_by_point_m_on_trend,
            # area for parameters not sure yet###################
            bar_count_to_by_point=self.p.bar_count_to_by_point,
            bar_cross_threshold=self.p.bar_cross_threshold,
            hl_length_threshold=self.p.hl_length_threshold,
            hl_overlap_ca_percent_multiplier=self.p.hl_overlap_ca_percent_multiplier,
            only_body_cross=self.p.only_body_cross,
            last_by_point_weight=self.p.last_by_point_weight,
            second_last_by_point_weight=self.p.second_last_by_point_weight,
            by_point_weight=self.p.by_point_weight,
            hl_extend_bar_cross_threshold=self.p.hl_extend_bar_cross_threshold,
            
        )
        self.viphl = VipHL( # entry point?
            close=self.data.close,
            open=self.data.open,
            high=self.data.high,
            low=self.data.low,
            mintick=self.p.mintick,
            bar_index=self.bar_index(),
            time=self.data.datetime,
            last_bar_index=self.last_bar_index(),
            settings=viphl_settings, # feeding parameters
            hls=[],
            vip_by_points=[],
            new_vip_by_points=[],
            recovery_windows=[],
            latest_recovery_windows={},
            normal_high_by_point=self.normal_high_by_point,
            normal_low_by_point=self.normal_low_by_point,
            trending_high_by_point=self.trending_high_by_point,
            trending_low_by_point=self.trending_low_by_point
            #########################################
        )

        # SMA for percentage change, can we directly average the percentage?
        self.close_average_percent = CloseAveragePercent(close_avg_percent_lookback=self.p.close_avg_percent_lookback)

        # For tracking trades
        self.trade_list = []
        self.result = {}

        # For pretty graph
        self.lines_info = []
        self.dl_lines_info = []

        # trade management
        self.max_equity: float = 0.0
        self.current_equity: float = 0.0
        self.max_drawdown: float = 0.0
        self.cur_drawdown: float = 0.0
        self.total_pnl: float = 0.0
        return

    def next(self):
        """
        moving along the timeline, evaluate the current status and trade(ish)
        """
        # conditions/status checking
        self.viphl.update_built_in_vars(bar_index=self.bar_index(), last_bar_index=self.last_bar_index())
        self.viphl.update(is_ma_trending=self.is_ma_trending, close_avg_percent=self.close_average_percent[0])


        # logic hidden?
        # Update the recovery window(站稳)
        self.viphl.update_recovery_window(
            trap_recover_window_threshold=self.params.trap_recover_window_threshold, # is this a percentage or # of bar?
            search_range=self.params.close_above_hl_search_range, # is this a percentage or # of bar?
            low_above_hl_threshold=self.params.low_above_hl_threshold, # is this a percentage or # of bar?
            close_avg_percent=self.close_average_percent[0] # is this a percentage or # of bar?
        )

        # Check the recovery window
        recovery_window_result = self.viphl.check_recovery_window_v3(
            close_avg_percent=self.close_average_percent[0],
            close_above_hl_threshold=self.params.close_above_hl_threshold,
            trap_recover_window_threshold=self.params.trap_recover_window_threshold,
            signal_window=self.params.signal_window,
            close_above_low_threshold=self.params.close_above_low_threshold,
            close_above_recover_low_threshold=self.params.close_above_recover_low_threshold,
            bar_count_close_above_hl_threshold=self.params.close_above_hl_bar_count,
            vvip_hl_min_by_point_count=self.params.vviphl_min_bypoint_count
        )

        # flattern RecoveryWindowResultV2
        flattern = from_recovery_window_result_v2(recovery_window_result)
        ######################################################################################

        break_hl_at_price: float = flattern.break_hl_at_price
        is_hl_satisfied: bool = flattern.is_hl_satisfied
        is_vvip_signal: bool = flattern.is_vvip_signal

        # Check stop loss
        quoted_trade = self.quote_trade()
        # Calculate stop loss thresholds
        # whats difference between normal and vviphl?
        stoploss_below_threshold = quoted_trade.stop_loss_percent < self.close_average_percent[0] * self.p.reduce_stop_loss_threshold
        vviphl_stoploss_below_threshold = quoted_trade.stop_loss_percent < self.close_average_percent[0] * self.p.vviphl_reduce_stop_loss_threshold


        # different between these three signals?

        ## trading signal
        within_lookback_period = self.last_bar_index() - self.bar_index() <= self.p.lookback

        # Normal signals
        has_long_signal = is_hl_satisfied and stoploss_below_threshold

        # VVIP signals
        has_vvip_long_signal = is_vvip_signal and vviphl_stoploss_below_threshold

        if within_lookback_period:# what is lookback period?
            if has_long_signal or has_vvip_long_signal:
                self.record_trade(0)
                self.viphl.commit_latest_recovery_window(break_hl_at_price)

        self.manage_trade()

    def quote_trade(self):
        # Calculate stop loss
        stop_loss_long = min(self.data.low[0], self.data.low[-1])
        stop_loss_percent = (self.data.close[0] - stop_loss_long) / self.data.close[0] * 100

        # Calculate HL byP scores for current market condition
        if self.is_ma_trending[0]:
            high_score = self.calculate_hl_byp_score(
                self.p.high_by_point_m_on_trend, 
                self.p.high_by_point_n_on_trend, 
                pivot_type='high', 
                is_trending=True
            )
            low_score = self.calculate_hl_byp_score(
                self.p.low_by_point_m_on_trend, 
                self.p.low_by_point_n_on_trend, 
                pivot_type='low', 
                is_trending=True
            )
        else:
            high_score = self.calculate_hl_byp_score(
                self.p.high_by_point_m, 
                self.p.high_by_point_n, 
                pivot_type='high', 
                is_trending=False
            )
            low_score = self.calculate_hl_byp_score(
                self.p.low_by_point_m, 
                self.p.low_by_point_n, 
                pivot_type='low', 
                is_trending=False
            )

        # Combined score for trade size adjustment
        combined_score = (high_score + low_score) / 2

        # Calculate entry size with scoring adjustment
        base_entry_size = math.floor(self.p.order_size_in_usd / self.data.close[0])
        entry_size = math.floor(base_entry_size * combined_score)

        # Create a new trade
        new_trade = TradeV2(
            entry_price=self.data.close[0],
            entry_time=self.data.datetime[0],
            entry_bar_index=self.bar_index(),
            entry_bar_offset=0,
            open_entry_size=entry_size,
            total_entry_size=entry_size,
            is_long=True,
            is_open=True,
            max_exit_price=self.data.high[0],
            stop_loss_percent=stop_loss_percent
        )

        return new_trade

    def record_trade(self, extend_bar_signal_offset):
        # Calculate HL byP scores for current market condition
        if self.is_ma_trending[0]:
            high_score = self.calculate_hl_byp_score(
                self.p.high_by_point_m_on_trend, 
                self.p.high_by_point_n_on_trend, 
                pivot_type='high', 
                is_trending=True
            )
            low_score = self.calculate_hl_byp_score(
                self.p.low_by_point_m_on_trend, 
                self.p.low_by_point_n_on_trend, 
                pivot_type='low', 
                is_trending=True
            )
        else:
            high_score = self.calculate_hl_byp_score(
                self.p.high_by_point_m, 
                self.p.high_by_point_n, 
                pivot_type='high', 
                is_trending=False
            )
            low_score = self.calculate_hl_byp_score(
                self.p.low_by_point_m, 
                self.p.low_by_point_n, 
                pivot_type='low', 
                is_trending=False
            )

        # Combined score for trade size adjustment
        combined_score = (high_score + low_score) / 2

        # Calculate entry size with scoring adjustment
        base_entry_size = math.floor(self.p.order_size_in_usd / self.data.close[0])
        entry_size = math.floor(base_entry_size * combined_score)

        self.trade_list.append(
            TradeV2(
                entry_price=self.data.close[0],
                entry_time=self.data.datetime[0],
                entry_bar_index=self.bar_index() - extend_bar_signal_offset,
                entry_bar_offset=0,
                open_entry_size=entry_size,
                total_entry_size=entry_size,
                is_long=True,
                is_open=True,
                max_exit_price=self.data.high[0]
            )
        )

    def manage_trade(self):
        self.cur_drawdown = 0.0

        # Calculate current drawdown
        for trade in self.trade_list:
            if trade.entry_time != self.data.datetime[0]:
                self.cur_drawdown += trade.open_entry_size * (trade.entry_price - self.data.low[0]) # data.high & low 是什么?
        self.cur_drawdown = self.max_equity - self.current_equity + self.cur_drawdown # 没太懂这个逻辑?
        self.max_drawdown = max(self.max_drawdown, self.cur_drawdown)

        # Process each trade
        for index, trade in enumerate(self.trade_list):
            cur_entry_time = trade.entry_time
            cur_entry_price = trade.entry_price
            if self.data.high[0] > trade.max_exit_price and self.data.datetime[0] > cur_entry_time: # 这里也不懂?
                trade.max_exit_price = self.data.high[0]

            trade.entry_bar_offset = self.bar_index() - trade.entry_bar_index
            stop_loss_long = min(self.data.low[-trade.entry_bar_offset], self.data.low[-(trade.entry_bar_offset + 1)])
            cur_max_return = (self.data.high[0] - cur_entry_price) / cur_entry_price * 100

            if trade.is_open:
                # Stop loss
                if trade.open_entry_size > 0 and self.data.close[0] < stop_loss_long and trade.is_long and trade.entry_time < self.data.datetime[0]:
                    trade.is_open = False
                    if self.p.toggle_pnl:
                        cur_return = (self.data.close[0] - cur_entry_price) / cur_entry_price * 100
                        if trade.take_profit:
                            trade.second_time = self.data.datetime[0]
                            if (trade.max_exit_price - cur_entry_price) / cur_entry_price * 100 > self.p.max_exit_ca_multiplier * self.close_average_percent:
                                cur_return = (trade.max_exit_price - cur_entry_price) * self.p.stop_gain_pt / cur_entry_price
                                self.exit_second_time(cur_entry_price, cur_return, trade)
                            else:
                                self.exit_second_time(cur_entry_price, cur_return, trade)
                        else:
                            self.exit_first_time(cur_entry_price, cur_return, trade)
                    trade.open_entry_size = 0

                # Reach 3M # why here? 3months?
                elif trade.entry_bar_offset == self.p.cycle_month * 20: # why 20, random value?
                    trade.is_open = False
                    cur_return = (trade.max_exit_price - cur_entry_price) / cur_entry_price * self.p.max_gain_pt
                    if self.p.toggle_pnl:
                        if trade.take_profit:
                            trade.second_time = self.data.datetime[0]
                            self.exit_second_time(cur_entry_price, cur_return, trade)
                        else:
                            self.exit_first_time(cur_entry_price, cur_return, trade)
                    trade.open_entry_size = 0

                # Take profit
                elif cur_max_return > max(self.p.first_gain_ca_multiplier * self.close_average_percent, self.p.stop_loss_pt) and not trade.take_profit and self.data.datetime[0] > trade.entry_time:
                    if self.p.toggle_pnl:
                        cur_return = max(self.p.first_gain_ca_multiplier * self.close_average_percent, self.p.stop_loss_pt)
                        trade.first_time = self.data.datetime[0]
                        trade.first_return = cur_return
                        self.current_equity += (1 + cur_return / 100) * cur_entry_price * int(trade.total_entry_size * 0.33)
                        self.max_equity = max(self.max_equity, self.current_equity)
                        self.total_pnl += cur_return / 3
                        trade.pnl += cur_return / 3
                    trade.take_profit = True # take ptofit meaning second time?
                    trade.open_entry_size -= int(trade.total_entry_size * 0.33)

                elif self.bar_index() == self.last_bar_index():
                    cur_return = (trade.max_exit_price - cur_entry_price) / cur_entry_price * self.p.max_gain_pt
                    if self.p.toggle_pnl:
                        if trade.take_profit:
                            self.exit_second_time(cur_entry_price, cur_return, trade)
                        else:
                            self.exit_first_time(cur_entry_price, cur_return, trade)
                    trade.open_entry_size = 0

        return self.cur_drawdown, self.max_equity, self.total_pnl

    def exit_first_time(self, cur_entry_price, cur_return, trade):
        trade.first_time = self.data.datetime[0]
        trade.first_return = cur_return
        self.current_equity += (1 + cur_return / 100) * cur_entry_price * trade.open_entry_size
        self.max_equity = max(self.max_equity, self.current_equity)
        self.total_pnl += cur_return
        trade.pnl += cur_return

    # why second time?
    def exit_second_time(self, cur_entry_price, cur_return, trade):
        trade.second_return = cur_return
        self.current_equity += (1 + cur_return / 100) * cur_entry_price * trade.open_entry_size
        self.max_equity = max(self.max_equity, self.current_equity)
        self.total_pnl += cur_return * 2 / 3
        trade.pnl += cur_return * 2 / 3

    def stop(self):
        self.export_hl_for_plotting()
        self.finalize_and_display_backtest_result()

    def export_hl_for_plotting(self):
        for hl in self.viphl.hls:
            if hl.extend_end_bar_index > 0 and hl.post_extend_end_bar_index == 0:
                end_index = hl.extend_end_bar_index
            elif hl.post_extend_end_bar_index > 0:
                end_index = hl.post_extend_end_bar_index
            else:
                end_index = hl.end_bar_index
            self.lines_info.append((hl.hl_value, hl.start_bar_index, end_index))

    def finalize_and_display_backtest_result(self):
        win_count = 0
        loss_count = 0
        win_pnl = 0.0
        loss_pnl = 0.0

        trade_detail_list = []
        for index, trade in enumerate(self.trade_list):
            if trade.pnl > 0:
                win_count += 1
                win_pnl += trade.pnl
            else:
                loss_count += 1
                loss_pnl += trade.pnl

            first_return = trade.first_return
            rounded_return = round(first_return, 2)
            formatted_return = f"{rounded_return:.2f}" if rounded_return == int(rounded_return) else str(rounded_return)

            second_return = str(round(trade.second_return, 2)) if trade.take_profit else ""
            second_trade_time = num2date(trade.second_time).date() if trade.second_time != 0 else self.data.close[0]

            cur_trade_detail = {
                "No.": index + 1,
                "Entry Date": f'{num2date(trade.entry_time).date()} {num2date(trade.entry_time).time()}',
                "Status:": "live" if trade.is_open else "closed",
                "Weighted Return:": round(trade.pnl, 2),
                "1st Trade Time": num2date(trade.first_time).date(),
                "1st Return%": formatted_return,
                "2nd Trade Time": second_trade_time if trade.take_profit else "",
                "2nd Return%": second_return
            }
            trade_detail_list.append(cur_trade_detail)

        total_pnl = round(self.total_pnl, 2)
        avg_pnl_per_entry = round(self.total_pnl / len(self.trade_list), 2) if len(self.trade_list) > 0 else 0
        trade_count = len(trade_detail_list)
        winning_entry_rate = round(win_count / trade_count * 100, 2) if trade_count > 0 else 0
        avg_winning_pnl = round(win_pnl / win_count, 2) if win_count > 0 else 0
        avg_losing_pnl = round(loss_pnl / loss_count, 2) if loss_count > 0 else 0
        fit_score = round(FIT_SCORE_MAX if loss_pnl == 0.0 else min((-win_pnl / loss_pnl), FIT_SCORE_MAX), 2)

        self.result = {
            "Total Pnl%": total_pnl,
            "Avg Pnl% per entry": avg_pnl_per_entry,
            "Trade Count": trade_count,
            "Winning entry%": winning_entry_rate,
            "Avg Winner%": avg_winning_pnl,
            "Avg Loser%": avg_losing_pnl,
            "Fit Score": fit_score
        }
        self.trade_detail_list = trade_detail_list

def load_data_from_csv(csv_file):
    # Read the CSV file
    dataframe = pd.read_csv(csv_file,
                                skiprows=0,
                                header=0,
                                parse_dates=True,
                                index_col=0)
    
    return dataframe

if __name__ == '__main__':
    dataframe = load_data_from_csv('BTC.csv')
    cerebro = bt.Cerebro()  # create a "Cerebro" engine instance

    # Create a data feed
    pandasData = bt.feeds.PandasData(dataname=dataframe)
    cerebro.adddata(pandasData)  # Add the data feed
    cerebro.broker.set_coc(True)

    # Add the trading strategy and set config
    cerebro.addstrategy(
        VipHLStrategy,
        mintick = 0.01,
        max_mn_cap = 100,  # HL byP scoring cap (adjustable)
        high_by_point_n= 10, # n is the # of bar on the left, m is right
        high_by_point_m= 10,
        low_by_point_n= 8,
        low_by_point_m= 8,
        high_by_point_n_on_trend= 5,
        high_by_point_m_on_trend= 5,
        low_by_point_n_on_trend= 4,
        low_by_point_m_on_trend= 4,
        # uncomment to change configurations
        # close_above_low_threshold=0.5,
        on_trend_ratio=1,
    )

    # extract results
    results = cerebro.run()
    strat = results[0]
    print(strat.result)

    # for plotting
    #cerebro.plot(style='candlestick', barup='green', bardown='red')
