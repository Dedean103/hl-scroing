#!/usr/bin/env python3
"""
Test script for the new high/low score scaling factors implementation
"""

# Test the scoring calculation directly
def test_scaling_factors():
    # Mock parameters
    class MockParams:
        def __init__(self):
            self.high_score_scaling_factor = 1.0
            self.low_score_scaling_factor = 1.0
    
    def calculate_combined_score(high_score, low_score, high_factor, low_factor):
        weighted_high = high_score * high_factor
        weighted_low = low_score * low_factor
        total_weight = high_factor + low_factor
        return (weighted_high + weighted_low) / total_weight
    
    # Test cases
    high_score = 0.5  # Normal high (10,10)
    low_score = 0.4   # Normal low (8,8)
    
    print("=== Scaling Factor Test ===")
    print(f"Base scores: High={high_score:.3f}, Low={low_score:.3f}")
    print()
    
    # Test 1: Equal weighting (default)
    combined = calculate_combined_score(high_score, low_score, 1.0, 1.0)
    print(f"Equal weighting (1.0, 1.0): {combined:.3f}")
    
    # Test 2: Emphasize low (support)
    combined = calculate_combined_score(high_score, low_score, 0.5, 1.0)
    print(f"Emphasize low (0.5, 1.0): {combined:.3f}")
    
    # Test 3: Emphasize high (resistance)
    combined = calculate_combined_score(high_score, low_score, 1.0, 0.5)
    print(f"Emphasize high (1.0, 0.5): {combined:.3f}")
    
    # Test 4: Strong low bias
    combined = calculate_combined_score(high_score, low_score, 0.2, 1.0)
    print(f"Strong low bias (0.2, 1.0): {combined:.3f}")
    
    print()
    print("Position size impact (40 BTC base):")
    base_size = 40
    
    for desc, h_factor, l_factor in [
        ("Equal weighting", 1.0, 1.0),
        ("Emphasize low", 0.5, 1.0), 
        ("Emphasize high", 1.0, 0.5),
        ("Strong low bias", 0.2, 1.0)
    ]:
        combined = calculate_combined_score(high_score, low_score, h_factor, l_factor)
        position = base_size * combined
        print(f"{desc:20}: {position:.1f} BTC ({combined:.1%})")

if __name__ == "__main__":
    test_scaling_factors()