from dataclasses import dataclass
from typing import List, Tuple
from indicators.common.base_indicator import BaseIndicator
from ..utils import calculate_weighted_hl_value, calculate_weighted_mn_values, get_dominant_mn

@dataclass
class HL(BaseIndicator):
    start_bar_index: int
    end_bar_index: int
    start_time: int
    end_time: int
    hl_value: float
    hl_accum_value: float
    by_point_values: List[float]  # array<float> in PineScript can be a List[float] in Python
    close_avg_percent: float
    bar_crosses: int = 0
    violated: bool = False  # Indicates if a HL is violated by body crosses
    extended: bool = False  # Indicates if a HL is extended to the first body cross
    extend_end_bar_index: int = 0
    bar_crosses_post_extend: int = 0
    historical_extend_bar_crosses_checked: bool = False  # Checked extend body cross for all historical bars post extend
    post_extend_end_bar_index: int = 0
    
    # NEW: mn tracking fields for trade scaling
    inducing_mn_values: List[Tuple[int, int]] = None  # Store (m,n) for each byPoint
    weighted_mn: Tuple[float, float] = (0.0, 0.0)    # Weighted average mn for this HL
    dominant_mn: Tuple[int, int] = (0, 0)             # Most significant mn (highest weight)

    def not_equal(self, target: 'HL') -> bool:
        return (
            self.start_bar_index != target.start_bar_index and
            self.end_bar_index != target.end_bar_index and
            self.start_time != target.start_time and
            self.end_time != target.end_time and
            self.hl_value != target.hl_value and
            self.hl_accum_value != target.hl_accum_value
        )

    def update_hl_value(self, last_by_point_weight: int, second_last_by_point_weight: int, by_point_weight: int):
        self.hl_value = calculate_weighted_hl_value(self.by_point_values, last_by_point_weight, second_last_by_point_weight, by_point_weight)

    def update_mn_values(self, last_by_point_weight: int, second_last_by_point_weight: int, by_point_weight: int):
        """
        Update weighted_mn and dominant_mn based on current inducing_mn_values.
        Should be called after any merge operation.
        """
        if self.inducing_mn_values is None:
            self.inducing_mn_values = []
            
        if len(self.inducing_mn_values) == 0:
            self.weighted_mn = (0.0, 0.0)
            self.dominant_mn = (0, 0)
        else:
            self.weighted_mn = calculate_weighted_mn_values(
                self.inducing_mn_values, 
                last_by_point_weight, 
                second_last_by_point_weight, 
                by_point_weight
            )
            self.dominant_mn = get_dominant_mn(self.inducing_mn_values)

    def merge(self, target: 'HL', last_by_point_weight: int, second_last_by_point_weight: int, by_point_weight: int, backward: bool) -> bool:
        is_not_equal = self.not_equal(target)

        if not backward:
            if is_not_equal:
                # Concatenate by_point_values
                self.by_point_values.extend(target.by_point_values)

                # NEW: Merge mn values
                if self.inducing_mn_values is None:
                    self.inducing_mn_values = []
                if target.inducing_mn_values is not None:
                    self.inducing_mn_values.extend(target.inducing_mn_values)

                # Update HL accumulative values
                self.hl_accum_value += target.hl_value
                self.update_hl_value(last_by_point_weight, second_last_by_point_weight, by_point_weight)
                
                # NEW: Update mn values after merge
                self.update_mn_values(last_by_point_weight, second_last_by_point_weight, by_point_weight)

                # Update end_bar_index and end_time
                self.end_bar_index = target.end_bar_index
                self.end_time = target.end_time

                # Reset extension-related attributes
                self.extended = False
                self.historical_extend_bar_crosses_checked = False
                self.bar_crosses_post_extend = 0
                self.post_extend_end_bar_index = 0
                self.extend_end_bar_index = 0
        else:
            if is_not_equal:
                # Prepend target by_point_values to source
                self.by_point_values = target.by_point_values + self.by_point_values

                # NEW: Merge mn values (prepend for backward merge)
                if self.inducing_mn_values is None:
                    self.inducing_mn_values = []
                if target.inducing_mn_values is not None:
                    self.inducing_mn_values = target.inducing_mn_values + self.inducing_mn_values

                # Update HL accumulative values
                self.hl_accum_value += target.hl_value
                self.update_hl_value(last_by_point_weight, second_last_by_point_weight, by_point_weight)
                
                # NEW: Update mn values after merge
                self.update_mn_values(last_by_point_weight, second_last_by_point_weight, by_point_weight)

                # Update start_bar_index and start_time
                self.start_bar_index = target.start_bar_index
                self.start_time = target.start_time

                # Reset extension-related attributes
                self.extended = False

        return True
    
    def overlap(self, target: 'HL', threshold: float) -> bool:
        """
        Check if the target hl_value overlaps with the source hl_value based on the given threshold.
        :param target: Another HL object
        :param threshold: Percentage threshold to define overlap range
        :return: True if overlap exists, False otherwise
        """
        higher_bound_y = self.hl_value * (1 + threshold / 100)
        lower_bound_y = self.hl_value * (1 - threshold / 100)

        # Check if target hl_value is within the bounds
        return lower_bound_y <= target.hl_value <= higher_bound_y
    
    def is_hl_bar_crosses_violated_if_merged(
        self, 
        end_hl: 'HL', 
        bar_cross_threshold: int, 
        last_by_point_weight: int, 
        second_last_by_point_weight: int, 
        by_point_weight: int, 
        current_cross_count: int
    ) -> bool:
        """
        Determines if the HL bar crosses are violated if merged.

        :param end_hl: The target HL object to merge with.
        :param bar_cross_threshold: The threshold of bar crosses allowed.
        :param last_by_point_weight: Weight for the last by point.
        :param second_last_by_point_weight: Weight for the second last by point.
        :param by_point_weight: The general by point weight.
        :param current_cross_count: Current number of bar crosses.
        :param bar_index: The current bar index.
        :param close: List of close prices.
        :param open_prices: List of open prices.
        :return: True if bar crosses are violated, otherwise False.
        """
        bar_cross_violated = False
        bar_count_to_loop = end_hl.end_bar_index - self.end_bar_index
        offset_to_start_hl_end_bar_index = self.bar_index - self.end_bar_index

        # Copy start and end HL by_point_values
        start_hl_by_points = self.by_point_values.copy()
        end_hl_by_points = end_hl.by_point_values.copy()

        # Calculate merged HL value
        merged_hl_value = calculate_weighted_hl_value(
            start_hl_by_points + end_hl_by_points, 
            last_by_point_weight, 
            second_last_by_point_weight, 
            by_point_weight
        )

        latest_cross_count = current_cross_count

        # Loop through bars to check cross counts
        for x in range(bar_count_to_loop):
            current_loop_index = offset_to_start_hl_end_bar_index - x
            if min(self.close[-current_loop_index], self.open[-current_loop_index]) < merged_hl_value \
                    and max(self.close[-current_loop_index], self.open[-current_loop_index]) > merged_hl_value:
                latest_cross_count += 1

        # Check if bar cross count exceeds the threshold
        if latest_cross_count > bar_cross_threshold:
            bar_cross_violated = True

        return bar_cross_violated
    
    # method isHLLengthPassedIfMerged(HL startHL, HL endHL, int lengthLimit) =>
    # endHL.endBarIndex - startHL.startBarIndex <= lengthLimit

    def is_hl_length_passed_if_merged(self, end_hl: 'HL', length_limit: int) -> bool:
        return end_hl.end_bar_index - self.start_bar_index <= length_limit

def swap(hls: List[HL], x_index: int, y_index: int):
    """
    Swap two elements in a list of HL objects at specified indices.
    :param hls: List of HL objects
    :param x_index: Index of the first element to swap
    :param y_index: Index of the second element to swap
    """
    # Ensure indices are within the range of the list
    if 0 <= x_index < len(hls) and 0 <= y_index < len(hls):
        # Swap elements at x_index and y_index
        hls[x_index], hls[y_index] = hls[y_index], hls[x_index]
    else:
        raise IndexError("Index out of range for the list of HL objects.")
