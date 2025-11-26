import math
from backtrader import DataSeries
from dataclasses import dataclass
from typing import List, Dict
from indicators.common.base_indicator import BaseIndicator
from .settings import Settings  # Assuming Settings class is defined in settings module
from .hl import HL  # Assuming HL class is defined in hl module
from .bypoint import ByPoint  # Assuming ByPoint class is defined in by_point module
from .recovery_window import RecoveryWindow, RecoveryWindowResult, RecoveryWindowFailure, RecoveryWindowSuccess, RecoveryWindowResultV2

@dataclass
class VipHL(BaseIndicator):
    settings: Settings
    hls: List[HL]  # List of HL objects
    vip_by_points: List[ByPoint]  # List of ByPoint objects for VIP
    new_vip_by_points: List[ByPoint]  # List of new VIP ByPoint objects
    recovery_windows: List[RecoveryWindow]  # List of RecoveryWindow objects
    normal_high_by_point: DataSeries
    normal_low_by_point: DataSeries
    trending_high_by_point: DataSeries
    trending_low_by_point: DataSeries
    latest_recovery_windows: Dict[float, int]

    def add_new_by_points_to_pending(self, is_ma_trending: DataSeries, close_avg_percent: float):
        new_by_point_found = False
        within_time_range = self.settings.debug_start_time <= self.time <= self.settings.debug_end_time
        maybe_only_draw_from_recent = True if not self.settings.draw_from_recent else (self.last_bar_index - self.bar_index < self.settings.bar_count_to_by_point)

        # # Normal self.high and self.low pivot points   
        normal_high_by_point = self.normal_high_by_point.value[0]
        normal_low_by_point = self.normal_low_by_point.value[0]

        # # Trending self.high and self.low pivot points
        trending_high_by_point = self.trending_high_by_point.value[0]
        trending_low_by_point = self.trending_low_by_point.value[0]

        # Check normal self.high pivot point
        if not math.isnan(normal_high_by_point) and not is_ma_trending[-self.settings.high_by_point_m]:
        # if normal_high_by_point is not None and not is_ma_trending[-self.settings.high_by_point_m]:
            within_price_range = self.settings.debug_start_price < normal_high_by_point < self.settings.debug_end_price
            if (not self.settings.debug or (within_time_range and within_price_range)) and maybe_only_draw_from_recent:
                new_by_point_found = True
                self.new_vip_by_points.append(
                    ByPoint(
                        n=self.settings.high_by_point_n,
                        m=self.settings.high_by_point_m,
                        price=normal_high_by_point,
                        close_at_pivot=self.close[-self.settings.high_by_point_m],
                        bar_index_at_pivot=self.bar_index - self.settings.high_by_point_m,
                        bar_time_at_pivot=self.time[0],
                        is_high=True,
                        is_trending=False,
                        used=False,
                        close_avg_percent=close_avg_percent
                    )
                )

        # Check normal self.low pivot point
        if not math.isnan(normal_low_by_point) and not is_ma_trending[-self.settings.low_by_point_m]:
            within_price_range = self.settings.debug_start_price < normal_low_by_point < self.settings.debug_end_price
            if (not self.settings.debug or (within_time_range and within_price_range)) and maybe_only_draw_from_recent:
                new_by_point_found = True
                self.new_vip_by_points.append(
                    ByPoint(
                        n=self.settings.low_by_point_n,
                        m=self.settings.low_by_point_m,
                        price=normal_low_by_point,
                        close_at_pivot=self.close[-self.settings.low_by_point_m],
                        bar_index_at_pivot=self.bar_index - self.settings.low_by_point_m,
                        bar_time_at_pivot=self.time[0],
                        is_high=False,
                        is_trending=False,
                        used=False,
                        close_avg_percent=close_avg_percent
                    )
                )

        # Check trending self.high pivot point
        if not math.isnan(trending_high_by_point) and is_ma_trending[-self.settings.high_by_point_m_on_trend]:
            within_price_range = self.settings.debug_start_price < trending_high_by_point < self.settings.debug_end_price
            if (not self.settings.debug or (within_time_range and within_price_range)) and maybe_only_draw_from_recent:
                new_by_point_found = True
                self.new_vip_by_points.append(
                    ByPoint(
                        n=self.settings.high_by_point_n_on_trend,
                        m=self.settings.high_by_point_m_on_trend,
                        price=trending_high_by_point,
                        close_at_pivot=self.close[-self.settings.high_by_point_m_on_trend],
                        bar_index_at_pivot=self.bar_index - self.settings.high_by_point_m_on_trend,
                        bar_time_at_pivot=self.time[0],
                        is_high=True,
                        is_trending=True,
                        used=False,
                        close_avg_percent=close_avg_percent
                    )
                )

        # Check trending self.low pivot point
        if not math.isnan(trending_low_by_point) and is_ma_trending[-self.settings.low_by_point_m_on_trend]:
            within_price_range = self.settings.debug_start_price < trending_low_by_point < self.settings.debug_end_price
            if (not self.settings.debug or (within_time_range and within_price_range)) and maybe_only_draw_from_recent:
                new_by_point_found = True
                self.new_vip_by_points.append(
                    ByPoint(
                        n=self.settings.low_by_point_n_on_trend,
                        m=self.settings.low_by_point_m_on_trend,
                        price=trending_low_by_point,
                        close_at_pivot=self.close[-self.settings.low_by_point_m_on_trend],
                        bar_index_at_pivot=self.bar_index - self.settings.low_by_point_m_on_trend,
                        bar_time_at_pivot=self.time[0],
                        is_high=False,
                        is_trending=True,
                        used=False,
                        close_avg_percent=close_avg_percent
                    )
                )

        return new_by_point_found
    
    def add_new_by_points_to_processed_array(self):
        while len(self.new_vip_by_points) > 0:
            self.vip_by_points.append(self.new_vip_by_points.pop())

    def sort_hls_by_end_bar_index(self):
        self.hls.sort(key=lambda hl: hl.end_bar_index)

        # -----below is original code------
        # hl_size = len(self.hls)

        # if hl_size > 1:
        #     for x in range(hl_size - 1):
        #         for y in range(x + 1, hl_size):
        #             line_x = self.hls[x]
        #             line_y = self.hls[y]
        #             if line_y.end_bar_index < line_x.end_bar_index:
        #                 # Swap the HL objects if line_y's end_bar_index is smaller than line_x's
        #                 self.hls[x], self.hls[y] = self.hls[y], self.hls[x]

    def remove_hl_by_end_bar_index(self):
        """
        Removes HLs from the beginning of the hls list if their end_bar_index plus 
        bar_count_to_by_point is less than the current bar_index.
        """
        do_shift = False
        
        # Check if the first HL in the list needs to be shifted
        if len(self.hls) > 0:
            if self.hls[0].end_bar_index + self.settings.bar_count_to_by_point < self.bar_index:
                do_shift = True

        # Continue removing HLs until the condition is no longer met
        while do_shift and len(self.hls) > 0:
            removed = self.hls.pop(0)  # Remove the first HL object

            # Update the shifting condition after each removal
            if len(self.hls) == 0 or self.hls[0].end_bar_index + self.settings.bar_count_to_by_point >= self.bar_index:
                do_shift = False

    def extend_hl_to_first_cross(self):
        """
        Extends each HL object in the list to the first valid cross after extension.
        """
        hl_size = len(self.hls)

        for x in range(hl_size):
            existing_hl = self.hls[x]

            # Calculate OHLC min and max
            ohlc_min = min(self.close, self.open, self.high, self.low)
            ohlc_max = max(self.close, self.open, self.high, self.low)
            body_min = min(self.close, self.open)
            body_max = max(self.close, self.open)
            bar_min = body_min if self.settings.only_body_cross else ohlc_min
            bar_max = body_max if self.settings.only_body_cross else ohlc_max

            # Handle first cross post extend
            if existing_hl.extended and not existing_hl.historical_extend_bar_crosses_checked and self.bar_index > existing_hl.extend_end_bar_index:
                if ohlc_min < existing_hl.hl_value < ohlc_max:
                    # Check if it's a valid cross
                    if bar_min < existing_hl.hl_value < bar_max:
                        existing_hl.bar_crosses_post_extend += 1

                    # Create the extension line
                    existing_hl.historical_extend_bar_crosses_checked = True
                    existing_hl.post_extend_end_bar_index = self.bar_index
                continue

            # Handle subsequent crosses after first extension
            if existing_hl.historical_extend_bar_crosses_checked:
                if ohlc_min < existing_hl.hl_value < ohlc_max:
                    if bar_min < existing_hl.hl_value < bar_max:
                        existing_hl.bar_crosses_post_extend += 1

                    # Update the extension line if threshold isn't violated
                    if existing_hl.bar_crosses_post_extend <= self.settings.hl_extend_bar_cross_threshold:
                        existing_hl.post_extend_end_bar_index = self.bar_index
                    else:
                        existing_hl.violated = True
                continue

            # Handle the first extension
            if (ohlc_min < existing_hl.hl_value < ohlc_max 
                and self.time != existing_hl.end_time):
                existing_hl.extended = True
                existing_hl.extend_end_bar_index = self.bar_index

    def sort_and_filter_hls(self):
        if self.bar_index == self.last_bar_index:
            self.sort_hls_by_end_bar_index()
            self.remove_hl_by_end_bar_index()

    def clear_all_hl(self):
        self.hls.clear()

    def rebuild_hl_from_most_recent_by_point(self, close_avg_percent: float = None):

        # Clear all existing HLs when new vip by point is found
        if len(self.new_vip_by_points) > 0:
            self.clear_all_hl()

        # Concatenate new VIP ByPoints with existing ones and clear new points
        self.vip_by_points.extend(self.new_vip_by_points)
        self.new_vip_by_points.clear()
        
        vip_by_point_size = len(self.vip_by_points)

        # Reverse the vip_by_points to loop from rightmost to left
        reversed_by_points = list(self.vip_by_points)[::-1]

        for x in range(vip_by_point_size):
            cur_new_by_point = reversed_by_points[x]

            # Create a new HL from the current ByPoint
            new_hl_from_by_point = HL(
                start_bar_index=cur_new_by_point.bar_index_at_pivot,
                end_bar_index=cur_new_by_point.bar_index_at_pivot,
                start_time=cur_new_by_point.bar_time_at_pivot,
                end_time=cur_new_by_point.bar_time_at_pivot,
                hl_value=cur_new_by_point.price,
                hl_accum_value=cur_new_by_point.price,
                by_point_values=[cur_new_by_point.price],
                close_avg_percent=cur_new_by_point.close_avg_percent,
                close=self.close,
                open=self.open,
                high=self.high,
                low=self.low,
                bar_index=self.bar_index,
                time=self.time,
                last_bar_index=self.last_bar_index,
                mintick=self.mintick,
                # NEW: Initialize mn values from the ByPoint
                inducing_mn_values=[(cur_new_by_point.m, cur_new_by_point.n)],
                weighted_mn=(float(cur_new_by_point.m), float(cur_new_by_point.n)),
                dominant_mn=(cur_new_by_point.m, cur_new_by_point.n)
            )

            is_new_hl = True

            # Loop through existing HLs to check for merging
            hl_size = len(self.hls)
            for y in range(hl_size):
                existing_hl = self.hls[y]

                # Check if there's an overlap
                if existing_hl.overlap(
                    new_hl_from_by_point, 
                    self.settings.hl_overlap_ca_percent_multiplier * existing_hl.close_avg_percent
                ):
                    # Check if there's no violation post-merge
                    no_bar_cross_violation_post_merge = not new_hl_from_by_point.is_hl_bar_crosses_violated_if_merged(
                        existing_hl, 
                        self.settings.bar_cross_threshold, 
                        self.settings.last_by_point_weight, 
                        self.settings.second_last_by_point_weight, 
                        self.settings.by_point_weight, 
                        existing_hl.bar_crosses
                    )

                    # Check if the HL length passes if merged
                    if no_bar_cross_violation_post_merge and new_hl_from_by_point.is_hl_length_passed_if_merged(
                        existing_hl, 
                        self.settings.hl_length_threshold
                    ):
                        existing_hl.merge(
                            new_hl_from_by_point, 
                            self.settings.last_by_point_weight, 
                            self.settings.second_last_by_point_weight, 
                            self.settings.by_point_weight,
                            backward=True
                        )
                        is_new_hl = False

                        # Break if reuse of ByPoint is not allowed
                        if not self.settings.allow_reuse_by_point:
                            break

            # If it's still a new HL, add to the list
            if is_new_hl:
                self.hls.append(new_hl_from_by_point)

    def extend_hl_to_first_cross_from_history(self):
        """
        Extend the high-low (HL) lines to the first cross from history, 
        checking for bar crosses and updating the HL status and lines.
        """

        hl_size = len(self.hls)

        for x in range(hl_size):
            existing_hl = self.hls[-x]
            bar_count_to_loop = self.bar_index - existing_hl.end_bar_index

            ohlc_min = min(self.close, self.open, self.high, self.low)
            ohlc_max = max(self.close, self.open, self.high, self.low)
            body_min = min(self.close, self.open)
            body_max = max(self.close, self.open)
            bar_min = body_min if self.settings.only_body_cross else ohlc_min
            bar_max = body_max if self.settings.only_body_cross else ohlc_max

            # If historical cross has been checked already
            if existing_hl.historical_extend_bar_crosses_checked:
                # If the current bar crosses the HL
                if ohlc_min < existing_hl.hl_value and ohlc_max > existing_hl.hl_value:
                    # Check if the cross is valid
                    if bar_min < existing_hl.hl_value and bar_max > existing_hl.hl_value:
                        existing_hl.bar_crosses_post_extend += 1

                    if existing_hl.bar_crosses_post_extend <= self.settings.hl_extend_bar_cross_threshold:
                        existing_hl.post_extend_end_bar_index = self.bar_index
                    else:
                        existing_hl.violated = True
                continue

            # Loop through historical bars for cross check
            for y in range(max(bar_count_to_loop - 1, 0), -1, -1):
                ohlc_min_y = min(self.close[-y], self.open[-y], self.high[-y], self.low[-y])
                ohlc_max_y = max(self.close[-y], self.open[-y], self.high[-y], self.low[-y])
                body_min_y = min(self.close[-y], self.open[-y])
                body_max_y = max(self.close[-y], self.open[-y])
                bar_min_y = body_min_y if self.settings.only_body_cross else ohlc_min_y
                bar_max_y = body_max_y if self.settings.only_body_cross else ohlc_max_y

                # If the HL hasn't been extended yet
                if not existing_hl.extended:
                    # Check for the first cross without counting it
                    if ohlc_min_y < existing_hl.hl_value and ohlc_max_y > existing_hl.hl_value and existing_hl.end_time < self.time[-y]:
                        existing_hl.extend_end_bar_index = self.bar_index - y
                        existing_hl.extended = True
                    continue

                # First cross after extension
                if existing_hl.extended and not existing_hl.historical_extend_bar_crosses_checked and (self.bar_index - y) > existing_hl.extend_end_bar_index:
                    # If the current bar crosses the HL
                    if ohlc_min_y < existing_hl.hl_value and ohlc_max_y > existing_hl.hl_value:
                        # Check if the cross is valid
                        if bar_min_y < existing_hl.hl_value and bar_max_y > existing_hl.hl_value:
                            existing_hl.bar_crosses_post_extend += 1

                        # Create a new line
                        existing_hl.post_extend_end_bar_index = self.bar_index - y
                        existing_hl.historical_extend_bar_crosses_checked = True
                    continue

                # Further crosses after historical extend check
                if existing_hl.historical_extend_bar_crosses_checked:
                    if ohlc_min_y < existing_hl.hl_value and ohlc_max_y > existing_hl.hl_value:
                        if bar_min_y < existing_hl.hl_value and bar_max_y > existing_hl.hl_value:
                            existing_hl.bar_crosses_post_extend += 1

                        # Update the line
                        if existing_hl.bar_crosses_post_extend <= self.settings.hl_extend_bar_cross_threshold:
                            existing_hl.post_extend_end_bar_index = self.bar_index - y
                        else:
                            existing_hl.violated = True

        return True
    
    def update_recovery_window(self, trap_recover_window_threshold, search_range, low_above_hl_threshold, close_avg_percent):
        """
        Updates the recovery windows for high-low (HL) based on specific thresholds.
        """
        hl_size = len(self.hls)

        for x in range(hl_size):
            cur_hl = self.hls[x]
            hl_value = cur_hl.hl_value
            close_below_hl = False
            break_but_close_above = False
            low_above_current_hl = False
            has_recovery_window = math.isclose(hl_value, self.round_to_mintick(hl_value)) in self.latest_recovery_windows

            # Count how many bars closed above HL
            bar_count_close_above_hl = 0
            for y in range(search_range, 0, -1):
                if self.close[-y] >= hl_value:
                    bar_count_close_above_hl += 1

            # Check conditions for recovery window
            if not (self.close[-1] < hl_value):
                if self.close < hl_value:
                    close_below_hl = True
                if self.close > hl_value and self.low < hl_value:
                    break_but_close_above = True
                if self.low > hl_value and hl_value * (1 + close_avg_percent * 0.01 * low_above_hl_threshold) >= self.low:
                    low_above_current_hl = True

            if close_below_hl or break_but_close_above or low_above_current_hl:
                new_window = RecoveryWindow(
                    break_hl_at_price=hl_value,
                    break_hl_at_bar_index=self.bar_index,
                    recover_at_bar_index=None,
                    break_hl_extend_bar_cross=cur_hl.bar_crosses_post_extend,
                    bar_count_close_above_hl=bar_count_close_above_hl,
                    vip_by_point_count=len(cur_hl.by_point_values),
                    # NEW: Pass mn values from the HL
                    hl_weighted_mn=cur_hl.weighted_mn,
                    hl_dominant_mn=cur_hl.dominant_mn
                )
                self.recovery_windows.append(new_window)

        # Remove recovery window if it's beyond the trapRecoverWindowThreshold
        do_shift = False
        if len(self.recovery_windows) > 0:
            if self.recovery_windows[0].break_hl_at_bar_index + trap_recover_window_threshold * 3 < self.bar_index:
                do_shift = True

        while do_shift:
            self.recovery_windows.pop(0)  # Shift the first element

            if len(self.recovery_windows) <= 0:
                do_shift = False

            if len(self.recovery_windows) > 0:
                if self.recovery_windows[0].break_hl_at_bar_index + trap_recover_window_threshold * 3 >= self.bar_index:
                    do_shift = False
                    

    def check_recovery_window(self, close_avg_percent, close_above_hl_threshold, trap_recover_window_threshold, 
                              signal_window, close_above_low_threshold, bar_count_close_above_hl_threshold):
        """
        Checks for recovery windows in the high-low (HL) based on various thresholds.
        """
        recovery_window_size = len(self.recovery_windows)
        has_signal = False
        close_above_low_and_hl = False
        violate_extend_bar_cross = False
        violate_recover_window = False
        violate_signal_window = False
        violate_search_range_close_above_bar_count = False
        recovery_window = None

        for x in range(recovery_window_size):
            cur_window = self.recovery_windows[x]
            hl_value = cur_window.break_hl_at_price
            has_recovery_window = math.isclose(hl_value, self.round_to_mintick(hl_value)) in self.latest_recovery_windows

            close_above_hl = self.close > cur_window.break_hl_at_price * (1 + close_avg_percent * 0.01 * close_above_hl_threshold)
            close_above_low = self.close > min(self.low, self.low[-1]) * (1 + close_avg_percent * 0.01 * close_above_low_threshold)

            # If recovery has already happened, skip this window
            if cur_window.recover_at_bar_index is not None:
                continue

            if close_above_hl and close_above_low:
                close_above_low_and_hl = True
                cur_window.recover_at_bar_index = self.bar_index

                # Check all validation
                pass_recover_window = cur_window.recover_at_bar_index - cur_window.break_hl_at_bar_index + 1 <= trap_recover_window_threshold
                pass_signal_window = not has_recovery_window or (self.bar_index - self.latest_recovery_windows.get(self.round_to_mintick(hl_value)) > signal_window)
                pass_search_range_close_above_hl = cur_window.bar_count_close_above_hl >= bar_count_close_above_hl_threshold
                pass_extend_bar_cross = cur_window.break_hl_extend_bar_cross <= self.settings.hl_extend_bar_cross_threshold
                pass_all_validation = pass_recover_window and pass_signal_window and pass_search_range_close_above_hl and pass_extend_bar_cross

                if pass_all_validation:
                    has_signal = True
                    violate_extend_bar_cross = False
                    violate_recover_window = False
                    violate_signal_window = False
                    violate_search_range_close_above_bar_count = False

                    recovery_window = cur_window.copy()
                    self.latest_recovery_windows[self.round_to_mintick(hl_value)] = self.bar_index
                    break
                else:
                    violate_extend_bar_cross = not pass_extend_bar_cross
                    violate_recover_window = not pass_recover_window
                    violate_signal_window = not pass_signal_window
                    violate_search_range_close_above_bar_count = not pass_search_range_close_above_hl

        return RecoveryWindowResult(
            has_signal,
            close_above_low_and_hl,
            violate_extend_bar_cross,
            violate_recover_window,
            violate_signal_window,
            violate_search_range_close_above_bar_count,
            recovery_window
        )
    
    def check_recovery_window_v3(self, close_avg_percent, close_above_hl_threshold, trap_recover_window_threshold, 
                             signal_window, close_above_low_threshold, close_above_recover_low_threshold, 
                             bar_count_close_above_hl_threshold, vvip_hl_min_by_point_count):
        """
        Checks recovery windows based on various thresholds and conditions.
        Translated from Pine Script's checkRecoveryWindowV3.
        """
        close_above_low_and_hl = False
        failures = []
        successes = []

        for cur_window in self.recovery_windows:
            hl_value = cur_window.break_hl_at_price
            has_recovery_window = self.round_to_mintick(hl_value) in self.latest_recovery_windows
            close_above_hl = self.close[0] > cur_window.break_hl_at_price * (1 + close_avg_percent * 0.01 * close_above_hl_threshold)
            close_above_low = self.close[0] > min(self.low[0], self.low[-1]) * (1 + close_avg_percent * 0.01 * close_above_low_threshold)

            # Skip if already recovered
            if cur_window.recovered:
                continue

            if close_above_hl:
                close_above_low_and_hl = True
                
                # Set recover_at_bar_index if not set
                if cur_window.recover_at_bar_index is None:
                    cur_window.recover_at_bar_index = self.bar_index

                recovery_bar_offset = self.bar_index - cur_window.recover_at_bar_index
                close_above_recovery_bar = self.close[0] > self.low[-recovery_bar_offset] * (1 + close_avg_percent * 0.01 * close_above_recover_low_threshold)

                signal_is_recover_bar = self.bar_index == cur_window.recover_at_bar_index
                signal_is_after_recover_bar = self.bar_index > cur_window.recover_at_bar_index
                signal_is_recover_bar_and_pass_two_day_low = signal_is_recover_bar and close_above_low
                signal_is_after_recover_bar_and_pass_recover_day_low = signal_is_after_recover_bar and close_above_recovery_bar

                if signal_is_recover_bar_and_pass_two_day_low or signal_is_after_recover_bar_and_pass_recover_day_low:
                    cur_window.recovered = True
                    
                    # Check all validation conditions
                    pass_recover_window = cur_window.recover_at_bar_index - cur_window.break_hl_at_bar_index + 1 <= trap_recover_window_threshold
                    pass_signal_window = not has_recovery_window or (self.bar_index - self.latest_recovery_windows.get(self.round_to_mintick(hl_value), 0) > signal_window)
                    pass_search_range_close_above_hl = cur_window.bar_count_close_above_hl >= bar_count_close_above_hl_threshold
                    pass_extend_bar_cross = cur_window.break_hl_extend_bar_cross <= self.settings.hl_extend_bar_cross_threshold
                    pass_all_validation = pass_recover_window and pass_signal_window and pass_search_range_close_above_hl and pass_extend_bar_cross

                    if pass_all_validation:
                        successes.append(RecoveryWindowSuccess(
                            has_signal=True,
                            is_vvip_signal=cur_window.vip_by_point_count >= vvip_hl_min_by_point_count,
                            recovery_window=cur_window.copy()
                        ))
                    else:
                        failures.append(RecoveryWindowFailure(
                            close_above_low_and_hl=close_above_low_and_hl,
                            violate_extend_bar_cross=not pass_extend_bar_cross,
                            violate_recover_window=not pass_recover_window,
                            violate_signal_window=not pass_signal_window,
                            violate_search_range_close_above_bar_count=not pass_search_range_close_above_hl,
                            recovery_window=cur_window.copy()
                        ))
            else:
                # Reset recover bar if close below
                cur_window.recover_at_bar_index = None

        # Return appropriate result based on successes and failures
        if not successes and not failures:
            return RecoveryWindowResultV2()
        elif not successes and failures:
            return RecoveryWindowResultV2(failure=failures[-1])
        elif successes:
            success = self.get_success_by_vvip(successes)
            return RecoveryWindowResultV2(success=success)
        else:
            return RecoveryWindowResultV2()
        
    def commit_latest_recovery_window(self, break_hl_at_price):
        rounded_price = self.round_to_mintick(break_hl_at_price)
        self.latest_recovery_windows[rounded_price] = self.bar_index

    def update(self, is_ma_trending: DataSeries, close_avg_percent: float) -> bool:
        """
        Update method for the VipHL class. Updates pivot points, constructs new HL objects, 
        sorts and filters the HLs, and extends them to the first cross.
        """
        # Update pivot points
        new_by_point_found = self.add_new_by_points_to_pending(is_ma_trending, close_avg_percent)

        if self.settings.draw_from_recent and self.last_bar_index - self.bar_index < self.settings.bar_count_to_by_point * 2:
            # 2. For each new by point
            if new_by_point_found:
                # 2.1. Every new by point is a starting point
                # 2.2. Loop through existing by points, and construct new HL
                self.rebuild_hl_from_most_recent_by_point(close_avg_percent)

            # Sort and filter HLs
            self.sort_and_filter_hls()

            self.extend_hl_to_first_cross_from_history()

        # Return true if new by point found
        return new_by_point_found
    
    def get_success_by_vvip(self, successes):
        size = len(successes)
        result = None

        for x in range(size):
            success = successes[x]
            if x == 0:
                result = success

            if success.is_vvip_signal:
                result = success
                break

        return result
