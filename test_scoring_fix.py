#!/usr/bin/env python3

"""
Test script to verify the HL byP scoring fix
Tests the scoring logic without requiring the full strategy dependencies
"""

class MockParams:
    def __init__(self):
        self.enable_hl_byp_scoring = True
        self.max_mn_cap = 20
        self.by_point_weight = 1
        self.on_trend_ratio = 1.5

def calculate_hl_byp_score_fixed(params, m, n, pivot_type='high', is_trending=False):
    '''Fixed version with condition-specific normalization (Option A)'''
    if not params.enable_hl_byp_scoring:
        return 1.0
    
    # Normalize window size to 0-1 range
    window_score = min((m + n) / (2 * params.max_mn_cap), 1.0)
    
    # Apply weight multipliers and condition-specific normalization (Option A)
    if is_trending:
        # Trending conditions are more significant with on_trend_ratio multiplier
        weight_multiplier = params.by_point_weight * params.on_trend_ratio
        max_possible_weight = params.by_point_weight * params.on_trend_ratio  # Trending-specific max
    else:
        # Normal conditions use base weight
        weight_multiplier = params.by_point_weight
        max_possible_weight = params.by_point_weight  # Normal-specific max
    
    # Final score incorporating weights (normalized to each condition's max)
    final_score = min(window_score * weight_multiplier / max_possible_weight, 1.0)
    
    print(f"[DEBUG] HL Score - Type: {pivot_type}, Trending: {is_trending}, "
          f"m+n: {m+n}, Window: {window_score:.3f}, Weight: {weight_multiplier:.3f}, "
          f"MaxWeight: {max_possible_weight:.3f}, Final: {final_score:.3f}")
    
    return final_score

def calculate_hl_byp_score_old(params, m, n, pivot_type='high', is_trending=False):
    '''Original flawed version for comparison'''
    if not params.enable_hl_byp_scoring:
        return 1.0
    
    # Normalize window size to 0-1 range
    window_score = min((m + n) / (2 * params.max_mn_cap), 1.0)
    
    # Apply weight multipliers based on pivot significance
    if is_trending:
        weight_multiplier = params.by_point_weight * params.on_trend_ratio
    else:
        weight_multiplier = params.by_point_weight
    
    # Final score incorporating weights (normalized back to 0-1) - FLAWED
    max_possible_weight = params.by_point_weight * params.on_trend_ratio  # Same for both!
    final_score = min(window_score * weight_multiplier / max_possible_weight, 1.0)
    
    return final_score

def test_scoring_scenarios():
    params = MockParams()
    
    print("=" * 80)
    print("TESTING HL byP SCORING FIX")
    print("=" * 80)
    print(f"Parameters: max_mn_cap={params.max_mn_cap}, on_trend_ratio={params.on_trend_ratio}")
    print()
    
    # Test scenarios
    scenarios = [
        ("Normal High", 10, 10, "high", False),
        ("Normal Low", 8, 8, "low", False), 
        ("Trending High", 5, 5, "high", True),
        ("Trending Low", 4, 4, "low", True),
    ]
    
    print("FIXED VERSION (Option A):")
    print("-" * 40)
    fixed_scores = {}
    for name, m, n, pivot_type, is_trending in scenarios:
        score = calculate_hl_byp_score_fixed(params, m, n, pivot_type, is_trending)
        fixed_scores[name] = score
        
    print()
    print("OLD VERSION (for comparison):")
    print("-" * 40)
    old_scores = {}
    for name, m, n, pivot_type, is_trending in scenarios:
        score = calculate_hl_byp_score_old(params, m, n, pivot_type, is_trending)
        old_scores[name] = score
        print(f"[OLD] {name}: {score:.3f}")
    
    print()
    print("COMBINED SCORES AND TRADE SIZE IMPACT:")
    print("-" * 40)
    
    # Calculate combined scores (as done in strategy)
    fixed_combined = (fixed_scores["Normal High"] + fixed_scores["Normal Low"]) / 2
    old_combined = (old_scores["Normal High"] + old_scores["Normal Low"]) / 2
    
    base_trade_size = 1000000  # $1M example
    
    print(f"Fixed Combined Score: {fixed_combined:.3f} → {fixed_combined*100:.1f}% trade size")
    print(f"Old Combined Score:   {old_combined:.3f} → {old_combined*100:.1f}% trade size")
    print(f"Improvement: {((fixed_combined/old_combined - 1) * 100):+.1f}%")
    
    print()
    print("VERIFICATION: on_trend_ratio=1 test (should show no trending bonus)")
    print("-" * 40)
    params.on_trend_ratio = 1.0
    
    no_bonus_normal = calculate_hl_byp_score_fixed(params, 10, 10, "high", False)
    no_bonus_trending = calculate_hl_byp_score_fixed(params, 5, 5, "high", True)
    
    print(f"Normal score: {no_bonus_normal:.3f}")
    print(f"Trending score: {no_bonus_trending:.3f}")
    print(f"Difference: {abs(no_bonus_normal - no_bonus_trending):.3f} (should be due to m+n difference only)")

if __name__ == "__main__":
    test_scoring_scenarios()