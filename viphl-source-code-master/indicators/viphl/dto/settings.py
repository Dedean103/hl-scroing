from dataclasses import dataclass

@dataclass
class Settings:
    high_by_point_n: int = 10
    high_by_point_m: int = 10
    low_by_point_n: int = 8
    low_by_point_m: int = 8

    high_by_point_n_on_trend: int = 5
    high_by_point_m_on_trend: int = 5
    low_by_point_n_on_trend: int = 4
    low_by_point_m_on_trend: int = 4

    bar_count_to_by_point: int = 700
    bar_cross_threshold: int = 5
    hl_length_threshold: int = 300

    hl_overlap_ca_percent_multiplier: float = 1.5
    
    only_body_cross: bool = True

    last_by_point_weight: int = 3
    second_last_by_point_weight: int = 2
    by_point_weight: int = 1
    hl_extend_bar_cross_threshold: int = 0  # Assuming default is 0 or needs to be set later

    draw_from_recent: bool = True
    allow_reuse_by_point: bool = False

    debug: bool = False
    debug_start_time: int = 0
    debug_end_time: int = 0
    debug_start_price: float = 0.0
    debug_end_price: float = 0.0
