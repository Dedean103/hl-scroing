def close_at_bar_top_by_percent(percent_in_decimal, close, low, high):
    return close > low + (high - low) * (1 - percent_in_decimal)


def greater_than_close_avg_by(close_avg_percent, multiplier, high, low):
    return ((high / low) - 1) > (close_avg_percent * multiplier * 0.01)


def close_at_bar_bottom_by_percent(percent_in_decimal, close, low, high):
    return close < low + (high - low) * percent_in_decimal
