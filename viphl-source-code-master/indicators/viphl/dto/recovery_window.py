from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class RecoveryWindow:
    break_hl_at_price: float 
    break_hl_at_bar_index: int
    recover_at_bar_index: Optional[int] = None
    break_hl_extend_bar_cross: int = 0
    bar_count_close_above_hl: int = 0
    vip_by_point_count: int = 0
    recovered: bool = False
    
    # NEW: mn values from the HL that created this recovery window
    hl_weighted_mn: Tuple[float, float] = (0.0, 0.0)
    hl_dominant_mn: Tuple[int, int] = (0, 0)

    def copy(self) -> 'RecoveryWindow':
        """Creates a copy of the RecoveryWindow instance"""
        return RecoveryWindow(
            break_hl_at_price=self.break_hl_at_price,
            break_hl_at_bar_index=self.break_hl_at_bar_index,
            recover_at_bar_index=self.recover_at_bar_index,
            break_hl_extend_bar_cross=self.break_hl_extend_bar_cross,
            bar_count_close_above_hl=self.bar_count_close_above_hl,
            vip_by_point_count=self.vip_by_point_count,
            recovered=self.recovered,
            # NEW: Copy mn values
            hl_weighted_mn=self.hl_weighted_mn,
            hl_dominant_mn=self.hl_dominant_mn
        )

@dataclass
class RecoveryWindowResult:
    has_signal: bool = False
    close_above_low_and_hl: bool = False  # New attribute added
    violate_extend_bar_cross: bool = False  # New attribute added
    violate_recover_window: bool = False  # New attribute added
    violate_signal_window: bool = False  # New attribute added
    violate_search_range_close_above_bar_count: bool = False  # New attribute added
    recovery_window: RecoveryWindow = None

@dataclass
class RecoveryWindowFailure:
    close_above_low_and_hl: bool = False
    violate_extend_bar_cross: bool = False
    violate_recover_window: bool = False
    violate_signal_window: bool = False
    violate_search_range_close_above_bar_count: bool = False
    recovery_window: Optional[RecoveryWindow] = None

@dataclass
class RecoveryWindowSuccess:
    has_signal: bool = False
    is_vvip_signal: bool = False
    signal_debug_line: Optional[str] = None
    recovery_window: Optional[RecoveryWindow] = None

@dataclass
class RecoveryWindowResultV2:
    failure: Optional[RecoveryWindowFailure] = None
    success: Optional[RecoveryWindowSuccess] = None

    def recovery_succeeded(self):
        return self.success is not None

    def recovery_failed(self):
        return self.failure is not None

    def recovery_has_result(self):
        return self.recovery_failed() or self.recovery_succeeded()

    def no_signal_but_close_above(self):
        if self.recovery_failed():
            return self.failure.close_above_low_and_hl
        return False

    def is_hl_satisfied(self):
        if self.recovery_succeeded():
            return self.success.has_signal
        return False

    def is_vvip_signal(self):
        if self.is_hl_satisfied():
            return self.success.is_vvip_signal
        return False

    def is_non_vvip_signal(self):
        if self.is_hl_satisfied():
            return not self.success.is_vvip_signal
        return False

    def violate_extended_bar_cross(self):
        if self.recovery_failed():
            return self.failure.violate_extend_bar_cross
        return False

    def violate_search_range_close_above_bar_count(self):
        if self.recovery_failed():
            return self.failure.violate_search_range_close_above_bar_count
        return False

    def violate_recover_window(self):
        if self.recovery_failed():
            return self.failure.violate_recover_window
        return False

    def violate_signal_window(self):
        if self.recovery_failed():
            return self.failure.violate_signal_window
        return False


def from_recovery_window_result_v2(result_v2: RecoveryWindowResultV2):

    if result_v2.recovery_succeeded():
        break_hl_at_price = result_v2.success.recovery_window.break_hl_at_price
    else:
        break_hl_at_price = -1

    return FlatternRecoveryWindowResultV2(
        break_hl_at_price=break_hl_at_price,
        is_hl_satisfied=result_v2.is_hl_satisfied(),
        is_vvip_signal=result_v2.is_vvip_signal(),
        is_non_vvip_signal=result_v2.is_non_vvip_signal(),
        no_signal_but_close_above=result_v2.no_signal_but_close_above(),
        violate_extended_bar_cross=result_v2.violate_extended_bar_cross(),
        violate_search_range_close_above_bar_count=result_v2.violate_search_range_close_above_bar_count(),
        violate_recover_window=result_v2.violate_recover_window(),
        violate_signal_window=result_v2.violate_signal_window()
    )


@dataclass
class FlatternRecoveryWindowResultV2:
    # this data class is a flattern version of
    # RecoveryWindowResultV2, which contain all the
    # required values for the main viphl 1.95 strategy code
    break_hl_at_price: float = -1

    is_hl_satisfied: bool = False
    is_vvip_signal: bool = False
    is_non_vvip_signal: bool = False
    no_signal_but_close_above: bool = False

    violate_extended_bar_cross: bool = False
    violate_search_range_close_above_bar_count: bool = False
    violate_recover_window: bool = False
    violate_signal_window: bool = False
