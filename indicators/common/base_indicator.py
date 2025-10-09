import math
from backtrader import DataSeries
from dataclasses import dataclass

@dataclass
class BaseIndicator:
    close: DataSeries
    open: DataSeries
    high: DataSeries
    low: DataSeries
    bar_index: int
    time: DataSeries
    last_bar_index: int
    mintick: float

    def update_built_in_vars(self, bar_index, last_bar_index):
        self.bar_index = bar_index
        self.last_bar_index = last_bar_index

    def round_to_mintick(self, value: float) -> float:
        """
        Rounds the value to the nearest value that can be divided by mintick, with ties rounding up.
        
        Args:
            value: The value to round
            mintick: The minimum tick size (e.g., 0.01 for 2 decimal places)
            
        Returns:
            float: The rounded value
        
        Examples:
            >>> round_to_mintick(10.123, 0.01) # Returns 10.12
            >>> round_to_mintick(10.125, 0.01) # Returns 10.13 (ties round up)
            >>> round_to_mintick(10.555, 0.05) # Returns 10.55
        """
        # Calculate how many minticks the value represents
        ticks = value / self.mintick
        
        # Round up if the fractional part is >= 0.5
        rounded_ticks = math.ceil(ticks) if (ticks % 1) >= 0.5 else math.floor(ticks)
        
        # Convert back to price by multiplying by mintick
        return rounded_ticks * self.mintick

