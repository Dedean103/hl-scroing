from dataclasses import dataclass

@dataclass
class ByPoint:
    n: int
    m: int
    price: float
    close_at_pivot: float
    bar_index_at_pivot: int  # index of the actual pivot
    bar_time_at_pivot: int
    is_high: bool
    is_trending: bool
    close_avg_percent: float
    used: bool = False
