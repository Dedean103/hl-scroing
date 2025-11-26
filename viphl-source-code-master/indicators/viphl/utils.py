from typing import List, Tuple

def calculate_weighted_hl_value(by_point_values: List[float], last_by_point_weight: int, second_last_by_point_weight: int, by_point_weight: int) -> float:
    by_point_count = len(by_point_values)
    norm = 0.0
    sum_val = 0.0

    if by_point_count == 0:
        return 0.0  # Handle empty list case

    for x in range(by_point_count):
        if x == by_point_count - 1:
            weight = last_by_point_weight
        elif x == by_point_count - 2:
            weight = second_last_by_point_weight
        else:
            weight = by_point_weight
        
        norm += weight
        sum_val += by_point_values[x] * weight

    return sum_val / norm if norm != 0 else 0.0  # Return weighted average, handle division by zero

def calculate_weighted_mn_values(mn_values: List[Tuple[int, int]], last_by_point_weight: int, second_last_by_point_weight: int, by_point_weight: int) -> Tuple[float, float]:
    """
    Calculate weighted average mn values based on ByPoint weights.
    More recent ByPoints have higher influence on final mn.
    
    Args:
        mn_values: List of (m, n) tuples from ByPoints
        last_by_point_weight: Weight for most recent ByPoint (typically 3)
        second_last_by_point_weight: Weight for second most recent (typically 2) 
        by_point_weight: Weight for older ByPoints (typically 1)
    
    Returns:
        Tuple[float, float]: Weighted average (m, n) values
    """
    mn_count = len(mn_values)
    
    if mn_count == 0:
        return (0.0, 0.0)  # Handle empty list case
    
    if mn_count == 1:
        return (float(mn_values[0][0]), float(mn_values[0][1]))  # Single value
    
    total_weight = 0.0
    weighted_m = 0.0
    weighted_n = 0.0

    for x in range(mn_count):
        if x == mn_count - 1:  # Most recent
            weight = last_by_point_weight
        elif x == mn_count - 2:  # Second most recent
            weight = second_last_by_point_weight
        else:  # Older ones
            weight = by_point_weight
        
        m, n = mn_values[x]
        weighted_m += m * weight
        weighted_n += n * weight
        total_weight += weight

    return (weighted_m / total_weight, weighted_n / total_weight) if total_weight > 0 else (0.0, 0.0)

def get_dominant_mn(mn_values: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Get the dominant (most recent/highest weighted) mn values.
    
    Args:
        mn_values: List of (m, n) tuples from ByPoints
    
    Returns:
        Tuple[int, int]: The most recent (highest weighted) mn values
    """
    if not mn_values:
        return (0, 0)
    
    # Most recent is at the end (highest weight)
    return mn_values[-1]
