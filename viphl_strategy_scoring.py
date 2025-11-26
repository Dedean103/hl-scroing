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
        ('enable_scoring_scale', True),      # Enable/disable scoring-scale system for position sizing and PnL scaling
        ('on_trend_ratio', 1.5),
        ('power_scaling_factor', 1.0),       # k: exponent for window scoring (m**k + n**k)
        ('high_score_scaling_factor', 1.0),  # Weight for high pivot contribution
        ('low_score_scaling_factor', 1.0),   # Weight for low pivot contribution

        # Dynamic mn Detection设置
        ('dynamic_mn_enabled', True),        # Enable/disable dynamic mn detection
        ('dynamic_mn_start', 4),            # Starting m=n value for dynamic detection
        ('dynamic_mn_step', 1),             # Increment step for dynamic detection (usually 1)

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
        ('debug_mode', False),  # Enable/disable debug printing
        ('debug_log_file', 'debug_log.txt'),  # File to save debug logs
        ('save_debug_to_file', True),  # Enable/disable saving debug to file
    )

    def log(self, txt, dt=None, doprint=True):
        ''' Logging function fot this strategy'''
        if doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def debug_log(self, message, to_console=True, to_file=None):
        """
        Enhanced debug logging that can output to both console and file
        
        Parameters:
        - message: The debug message to log
        - to_console: Whether to print to console (default: True)  
        - to_file: Whether to save to file (default: uses save_debug_to_file parameter)
        """
        # Determine file logging preference
        if to_file is None:
            to_file = self.p.save_debug_to_file
            
        # If neither console nor file output is enabled, return early
        if not (self.p.debug_mode and to_console) and not to_file:
            return
            
        # Add timestamp
        if hasattr(self, 'data') and len(self.data) > 0:
            timestamp = f"[{self.data.datetime.date(0)} {self.data.datetime.time(0)}] "
        else:
            from datetime import datetime
            timestamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        
        formatted_message = timestamp + message
        
        # Console output (only if debug_mode is True)
        if self.p.debug_mode and to_console:
            print(formatted_message)
        
        # File output (can work independently of debug_mode)
        if to_file and self.p.debug_log_file:
            try:
                with open(self.p.debug_log_file, 'a', encoding='utf-8') as f:
                    f.write(formatted_message + '\n')
            except Exception as e:
                print(f"Warning: Could not write to debug file {self.p.debug_log_file}: {e}")

    def init_debug_log_file(self):
        """Initialize/clear the debug log file at the start of strategy"""
        if self.p.save_debug_to_file and self.p.debug_log_file:
            try:
                with open(self.p.debug_log_file, 'w', encoding='utf-8') as f:
                    f.write("=== VipHL Strategy Debug Log ===\n")
                    f.write(f"Strategy started at: {self.data.datetime.date(0) if hasattr(self, 'data') and len(self.data) > 0 else 'Unknown'}\n")
                    f.write(f"Dynamic mn enabled: {self.p.dynamic_mn_enabled}\n")
                    if self.p.dynamic_mn_enabled:
                        f.write(f"Dynamic mn range: {self.p.dynamic_mn_start} to {self.p.max_mn_cap}\n")
                    f.write("=" * 50 + "\n\n")
                print(f"Debug log initialized: {self.p.debug_log_file}")
            except Exception as e:
                print(f"Warning: Could not initialize debug file {self.p.debug_log_file}: {e}")

    def calculate_hl_byp_score(self, m, n, pivot_type='high', is_trending=False):
        '''Calculate normalized HL byP score (0-1) based on m,n parameters with power scaling'''
        if not self.p.enable_hl_byp_scoring:
            return 1.0

        # Apply power scaling factor k to window size calculation
        k = self.p.power_scaling_factor
        # Normalize window size to 0-1 range using power scaling: (m**k + n**k) / (2 * max_mn_cap**k)
        window_score = min((m**k + n**k) / (2 * (self.p.max_mn_cap ** k)), 1.0)

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
        if (self.p.debug_mode or self.p.save_debug_to_file) and hasattr(self, 'data') and len(self.data) > 0:
            debug_msg = (f"HL Score - Type: {pivot_type}, Trending: {is_trending}, "
                        f"k: {k:.2f}, m+n: {m+n}, m^k+n^k: {m**k + n**k:.3f}, "
                        f"Window: {window_score:.3f}, Weight: {weight_multiplier:.3f}, "
                        f"MaxWeight: {max_possible_weight:.3f}, Final: {final_score:.3f}")
            self.debug_log(debug_msg)

        return final_score

    def calculate_position_scale(self, combined_score):
        """Map combined_score to a bounded 1x-3x scaling factor."""
        scale = 1.0 + 2.0 * combined_score
        return max(1.0, min(scale, 3.0))

    def validate_pivot_with_mn(self, bar_index, m, n, pivot_type):
        """Validate if a bar is a pivot with the supplied m/n window."""
        # Check bounds - need n bars on left and m bars on right
        # len(self.data) only reflects processed bars. Use the full buffer
        # length via last_bar_index so future bars are accessible for pivot
        # validation in backtests.
        max_index = self.last_bar_index()
        if (bar_index - n < 0) or (bar_index + m > max_index):
            return False
        
        # Absolute indexing is easier through the cached numpy arrays
        high_array = self.data.high.array
        low_array = self.data.low.array
        
        if pivot_type == 'high':
            center_price = high_array[bar_index]
            
            # Check left side: all bars must be lower than center
            for i in range(bar_index - n, bar_index):
                if high_array[i] >= center_price:
                    return False
                    
            # Check right side: all bars must be lower than center  
            for i in range(bar_index + 1, bar_index + m + 1):
                if high_array[i] >= center_price:
                    return False
                    
        else:  # low pivot
            center_price = low_array[bar_index]
            
            # Check left side: all bars must be higher than center
            for i in range(bar_index - n, bar_index):
                if low_array[i] <= center_price:
                    return False
                    
            # Check right side: all bars must be higher than center
            for i in range(bar_index + 1, bar_index + m + 1):
                if low_array[i] <= center_price:
                    return False
        
        return True
    def find_dynamic_pivot(self, bar_index, pivot_type='high'):
        """
        Progressive pivot validation starting from dynamic_mn_start up to max_mn_cap
        Returns (best_m, best_n, pivot_confirmed) or (None, None, False)
        """
        if not self.p.dynamic_mn_enabled:
            # Fall back to original static behavior
            if pivot_type == 'high':
                m, n = self.p.high_by_point_m, self.p.high_by_point_n
                if self.is_ma_trending[0]:
                    m, n = self.p.high_by_point_m_on_trend, self.p.high_by_point_n_on_trend
            else:
                m, n = self.p.low_by_point_m, self.p.low_by_point_n
                if self.is_ma_trending[0]:
                    m, n = self.p.low_by_point_m_on_trend, self.p.low_by_point_n_on_trend
            
            is_valid = self.validate_pivot_with_mn(bar_index, m, n, pivot_type)
            return (m, n, is_valid)
        
        best_m = best_n = None
        tested_windows = []
        
        # Progressive validation from start to max
        for window_size in range(self.p.dynamic_mn_start, 
                               self.p.max_mn_cap + 1, 
                               self.p.dynamic_mn_step):
            m = n = window_size
            is_valid = self.validate_pivot_with_mn(bar_index, m, n, pivot_type)
            tested_windows.append((m, n, is_valid))
            
            if is_valid:
                best_m, best_n = m, n  # Keep expanding
            else:
                break  # Failed at this size, stop
        
        pivot_found = (best_m is not None)
        
        # Comprehensive logging
        if (self.p.debug_mode or self.p.save_debug_to_file) and hasattr(self, 'data') and len(self.data) > 0:
            date_str = self.data.datetime.date(0).isoformat()
            time_str = self.data.datetime.time(0).isoformat()
            price = self.data.high[0] if pivot_type == 'high' else self.data.low[0]
            trending = "TRENDING" if self.is_ma_trending[0] else "NORMAL"
            
            # Main pivot detection header
            header_msg = f"[DYNAMIC-{pivot_type.upper()}] Bar {bar_index} | {date_str} {time_str} | Price: {price:.2f} | {trending}"
            self.debug_log(header_msg)
            
            # Test results for each mn value
            for m, n, valid in tested_windows:
                status = "✓ PASS" if valid else "✗ FAIL"
                test_msg = f"  mn={m:2d} -> {status}"
                self.debug_log(test_msg)
            
            # Final result
            if pivot_found:
                result_msg = f"  🎯 PIVOT CONFIRMED with mn={best_m} | Score will use m={best_m}, n={best_n}"
            else:
                result_msg = f"  ❌ NO PIVOT FOUND"
            self.debug_log(result_msg)
            self.debug_log("")  # Add blank line for readability
        
        return (best_m, best_n, pivot_found)

    def get_trade_mn_values(self, recovery_window_result):
        """Get mn values specific to the HL that triggered this trade"""
        if recovery_window_result.recovery_succeeded():
            recovery_window = recovery_window_result.success.recovery_window
            
            # Use the mn from the HL that created this recovery window
            hl_mn = recovery_window.hl_weighted_mn
            
            # For now, use the same mn for both high and low
            # This could be enhanced later to track high/low specific mn values
            return hl_mn, hl_mn
        else:
            # Fallback to current logic if no recovery window
            return self.get_current_mn_values()

    def get_current_mn_values(self):
        """
        Get the current m,n values to use for scoring.
        If dynamic detection is enabled, use the current bar's dynamic values.
        Otherwise, fall back to static parameters.
        """
        if self.p.dynamic_mn_enabled:
            current_bar = self.bar_index()
            
            # Get dynamic mn for high pivots
            high_m, high_n, high_found = self.find_dynamic_pivot(current_bar, 'high')
            if not high_found:
                # Fall back to static values if no dynamic pivot found
                if self.is_ma_trending[0]:
                    high_m, high_n = self.p.high_by_point_m_on_trend, self.p.high_by_point_n_on_trend
                else:
                    high_m, high_n = self.p.high_by_point_m, self.p.high_by_point_n
            
            # Get dynamic mn for low pivots
            low_m, low_n, low_found = self.find_dynamic_pivot(current_bar, 'low')
            if not low_found:
                # Fall back to static values if no dynamic pivot found
                if self.is_ma_trending[0]:
                    low_m, low_n = self.p.low_by_point_m_on_trend, self.p.low_by_point_n_on_trend
                else:
                    low_m, low_n = self.p.low_by_point_m, self.p.low_by_point_n
            
            return (high_m, high_n), (low_m, low_n)
        else:
            # Static behavior
            if self.is_ma_trending[0]:
                high_mn = (self.p.high_by_point_m_on_trend, self.p.high_by_point_n_on_trend)
                low_mn = (self.p.low_by_point_m_on_trend, self.p.low_by_point_n_on_trend)
            else:
                high_mn = (self.p.high_by_point_m, self.p.high_by_point_n)
                low_mn = (self.p.low_by_point_m, self.p.low_by_point_n)
            
            return high_mn, low_mn

    def __init__(self):
        '''init is called once at the last row'''

        # Validate scaling factors
        if self.p.high_score_scaling_factor <= 0:
            raise ValueError(f"high_score_scaling_factor must be positive, got {self.p.high_score_scaling_factor}")
        if self.p.low_score_scaling_factor <= 0:
            raise ValueError(f"low_score_scaling_factor must be positive, got {self.p.low_score_scaling_factor}")

        # Initialize debug logging
        if self.p.save_debug_to_file:
            self.init_debug_log_file()

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

        # Dynamic pivot detection (additional layer)
        if self.p.dynamic_mn_enabled:
            current_bar = self.bar_index()
            
            # Test for high pivot with dynamic mn
            high_m, high_n, high_found = self.find_dynamic_pivot(current_bar, 'high')
            if high_found:
                high_score = self.calculate_hl_byp_score(high_m, high_n, 'high', self.is_ma_trending[0])
                if self.p.debug_mode or self.p.save_debug_to_file:
                    self.debug_log(f"  📊 HIGH PIVOT SCORE: {high_score:.4f} (m={high_m}, n={high_n})")
            
            # Test for low pivot with dynamic mn  
            low_m, low_n, low_found = self.find_dynamic_pivot(current_bar, 'low')
            if low_found:
                low_score = self.calculate_hl_byp_score(low_m, low_n, 'low', self.is_ma_trending[0])
                if self.p.debug_mode or self.p.save_debug_to_file:
                    self.debug_log(f"  📊 LOW PIVOT SCORE: {low_score:.4f} (m={low_m}, n={low_n})")

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

        # Get current mn values (dynamic or static)
        (high_m, high_n), (low_m, low_n) = self.get_current_mn_values()

        # Calculate HL byP scores using the determined m,n values
        high_score = self.calculate_hl_byp_score(
            high_m, high_n, 
            pivot_type='high', 
            is_trending=self.is_ma_trending[0]
        )
        low_score = self.calculate_hl_byp_score(
            low_m, low_n, 
            pivot_type='low', 
            is_trending=self.is_ma_trending[0]
        )

        # Combined score for trade size adjustment with scaling factors
        weighted_high = high_score * self.p.high_score_scaling_factor
        weighted_low = low_score * self.p.low_score_scaling_factor
        total_weight = self.p.high_score_scaling_factor + self.p.low_score_scaling_factor
        combined_score = (weighted_high + weighted_low) / total_weight

        # Debug logging for weighted scoring
        if (self.p.debug_mode or self.p.save_debug_to_file) and hasattr(self, 'data') and len(self.data) > 0:
            debug_msg = (f"Combined Score - High: {high_score:.3f}*{self.p.high_score_scaling_factor:.1f}={weighted_high:.3f}, "
                        f"Low: {low_score:.3f}*{self.p.low_score_scaling_factor:.1f}={weighted_low:.3f}, "
                        f"Combined: {combined_score:.3f}")
            self.debug_log(debug_msg)

        # Calculate entry size with scoring adjustment
        base_entry_size = max(1, math.floor(self.p.order_size_in_usd / self.data.close[0]))
        if self.p.enable_scoring_scale:
            size_scale = self.calculate_position_scale(combined_score)
            entry_size = max(1, math.floor(base_entry_size * size_scale))
        else:
            size_scale = 1.0
            entry_size = base_entry_size

        if (self.p.debug_mode or self.p.save_debug_to_file) and hasattr(self, 'data') and len(self.data) > 0:
            self.debug_log(f"Entry Size - Base: {base_entry_size}, Scale: {size_scale:.3f}, Final: {entry_size}")

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
        # Get the latest recovery window result to extract HL-specific mn values
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

        # Use HL-specific mn values instead of current detection
        (high_m, high_n), (low_m, low_n) = self.get_trade_mn_values(recovery_window_result)

        # Calculate HL byP scores using the HL-specific m,n values
        high_score = self.calculate_hl_byp_score(
            high_m, high_n,
            pivot_type='high',
            is_trending=self.is_ma_trending[0]
        )
        low_score = self.calculate_hl_byp_score(
            low_m, low_n,
            pivot_type='low',
            is_trending=self.is_ma_trending[0]
        )

        # Combined score for trade size adjustment with scaling factors
        weighted_high = high_score * self.p.high_score_scaling_factor
        weighted_low = low_score * self.p.low_score_scaling_factor
        total_weight = self.p.high_score_scaling_factor + self.p.low_score_scaling_factor
        combined_score = (weighted_high + weighted_low) / total_weight

        # Debug logging for weighted scoring
        if (self.p.debug_mode or self.p.save_debug_to_file) and hasattr(self, 'data') and len(self.data) > 0:
            debug_msg = (f"HL-Trade Scaling - mn_high: ({high_m:.1f},{high_n:.1f}), mn_low: ({low_m:.1f},{low_n:.1f}), "
                        f"High: {high_score:.3f}*{self.p.high_score_scaling_factor:.1f}={weighted_high:.3f}, "
                        f"Low: {low_score:.3f}*{self.p.low_score_scaling_factor:.1f}={weighted_low:.3f}, "
                        f"Combined: {combined_score:.3f}")
            self.debug_log(debug_msg)

        # Calculate entry size using bounded scaling (1x - 3x)
        size_scale = self.calculate_position_scale(combined_score) if self.p.enable_scoring_scale else 1.0
        base_entry_size = max(1, math.floor(self.p.order_size_in_usd / self.data.close[0]))
        entry_size = max(1, math.floor(base_entry_size * size_scale))

        if (self.p.debug_mode or self.p.save_debug_to_file) and hasattr(self, 'data') and len(self.data) > 0:
            self.debug_log(f"Entry Size - Base: {base_entry_size}, Scale: {size_scale:.3f}, Final: {entry_size}")

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
                max_exit_price=self.data.high[0],
                # NEW: Store the HL-specific mn values for trade lifecycle consistency
                trade_mn_high=(high_m, high_n),
                trade_mn_low=(low_m, low_n),
                original_combined_score=combined_score
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
                        # Record partial exit PnL based on actual sized position
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
        max_mn_cap = 20,  # HL byP scoring cap (also max for dynamic mn)
        
        # Dynamic mn settings
        dynamic_mn_enabled = True,   # Enable dynamic mn detection
        dynamic_mn_start = 4,        # Start testing from mn=4
        dynamic_mn_step = 1,         # Increment by 1
        
        # Static mn settings (fallback values)
        high_by_point_n= 10, # n is the # of bar on the left, m is right
        high_by_point_m= 10,
        low_by_point_n= 10,
        low_by_point_m= 10,
        high_by_point_n_on_trend= 10,
        high_by_point_m_on_trend= 10,
        low_by_point_n_on_trend= 10,
        low_by_point_m_on_trend= 10,
        
        power_scaling_factor= 1.0,      # k: 1.0 (linear), >1 (exponential), <1 (diminishing)
        high_score_scaling_factor= 0.5,  # Weight for high pivot contribution
        low_score_scaling_factor= 0.5,   # Weight for low pivot contribution
        
        # Debug logging settings
        debug_mode=True,  # Enable debug printing to see dynamic mn detection
        save_debug_to_file=True,  # Save debug logs to file
        debug_log_file="dynamic_mn_debug.txt",  # Debug log filename
        
        # uncomment to change configurations
        # close_above_low_threshold=0.5,
        on_trend_ratio=1,
    )

    # extract results
    results = cerebro.run()
    strat = results[0]
    print(strat.result)

    # for plotting
    cerebro.plot(style='candlestick', barup='green', bardown='red')
