from dataclasses import dataclass
from typing import Tuple

@dataclass
class TradeV2:
    entry_price: float
    entry_time: int
    entry_bar_index: int  # Note that for DL window signal, entry bar index is the same as the original signal bar index
    entry_bar_offset: int

    open_entry_size: int
    total_entry_size: int
    is_long: bool
    is_open: bool

    max_exit_price: float
    take_profit: bool = False
    pnl: float = 0.0

    first_time: int = 0
    first_return: float = 0.0
    second_time: int = 0
    second_return: float = 0.0
    stop_loss_percent: float = 0.0
    
    # NEW: Store the mn used for this trade's scaling
    trade_mn_high: Tuple[float, float] = (0.0, 0.0)
    trade_mn_low: Tuple[float, float] = (0.0, 0.0)
    original_combined_score: float = 0.0
