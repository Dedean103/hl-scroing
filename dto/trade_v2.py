from dataclasses import dataclass

@dataclass
class TradeV2:
    entry_price: float
    entry_time: int
    entry_bar_index: int  # Note that for DL window signal, entry bar index is the same as the original signal bar index
    entry_bar_offset: int

    open_entry_size: float
    total_entry_size: float
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
    combined_score: float = 0.0
    high_m: float = 0.0
    high_n: float = 0.0
    low_m: float = 0.0
    low_n: float = 0.0
    is_trending_trade: bool = False
    high_source: str = ""
    low_source: str = ""
