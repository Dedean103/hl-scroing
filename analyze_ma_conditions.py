import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('BTC.csv', parse_dates=True, index_col=0)

# Calculate moving averages
df['MA10'] = df['close'].rolling(window=10).mean()
df['MA40'] = df['close'].rolling(window=40).mean()
df['MA100'] = df['close'].rolling(window=100).mean()

# Calculate MA delta (trending condition)
df['MA_Delta'] = df['MA10'] - df['MA100']

# Calculate percentile of MA delta over 500-bar lookback (matching strategy config)
df['MA_Delta_Percentile'] = df['MA_Delta'].rolling(window=500).apply(
    lambda x: np.percentile(x, 1) if len(x) == 500 else np.nan
)

# Check trending conditions
df['MA_Greater'] = (df['MA10'] > df['MA40']) & (df['MA40'] > df['MA100'])
df['MA_Trending'] = df['MA_Greater'] & (df['MA_Delta'] > df['MA_Delta_Percentile'])

# Print summary statistics
print("=" * 80)
print("MOVING AVERAGE TRENDING ANALYSIS")
print("=" * 80)
print(f"\nTotal bars in dataset: {len(df)}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Find first occurrence of trending condition
trending_bars = df[df['MA_Trending'] == True]
if len(trending_bars) > 0:
    first_trending = trending_bars.index[0]
    print(f"\n[YES] First trending condition met: {first_trending}")
    print(f"   Price at that time: ${df.loc[first_trending, 'close']:.2f}")
    print(f"   MA10: ${df.loc[first_trending, 'MA10']:.2f}")
    print(f"   MA40: ${df.loc[first_trending, 'MA40']:.2f}")
    print(f"   MA100: ${df.loc[first_trending, 'MA100']:.2f}")
else:
    print("\n[NO] No trending condition met in the dataset")

# Analyze by year
print("\n" + "=" * 80)
print("TRENDING CONDITIONS BY YEAR")
print("=" * 80)
for year in range(2020, 2026):
    year_data = df[df.index.year == year]
    if len(year_data) > 0:
        ma_greater_pct = (year_data['MA_Greater'].sum() / len(year_data)) * 100
        ma_trending_pct = (year_data['MA_Trending'].sum() / len(year_data)) * 100
        avg_price = year_data['close'].mean()
        print(f"\n{year}:")
        print(f"  Avg Price: ${avg_price:.2f}")
        print(f"  MA Bullish Alignment (MA10>MA40>MA100): {ma_greater_pct:.1f}%")
        print(f"  MA Trending (+ percentile check): {ma_trending_pct:.1f}%")

# Create visualization
fig, axes = plt.subplots(4, 1, figsize=(15, 12))

# Plot 1: Price and MAs
ax1 = axes[0]
ax1.plot(df.index, df['close'], label='Close Price', linewidth=1.5, color='blue')
ax1.plot(df.index, df['MA10'], label='MA10', linewidth=1, alpha=0.7)
ax1.plot(df.index, df['MA40'], label='MA40', linewidth=1, alpha=0.7)
ax1.plot(df.index, df['MA100'], label='MA100', linewidth=1, alpha=0.7)
ax1.set_ylabel('Price ($)')
ax1.set_title('BTC Price with Moving Averages')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: MA Delta
ax2 = axes[1]
ax2.plot(df.index, df['MA_Delta'], label='MA Delta (MA10 - MA100)', color='purple')
ax2.plot(df.index, df['MA_Delta_Percentile'], label='1st Percentile Threshold',
         color='red', linestyle='--', alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.fill_between(df.index, df['MA_Delta'], 0, where=(df['MA_Delta'] > 0),
                  alpha=0.3, color='green', label='Positive Delta')
ax2.fill_between(df.index, df['MA_Delta'], 0, where=(df['MA_Delta'] < 0),
                  alpha=0.3, color='red', label='Negative Delta')
ax2.set_ylabel('Delta ($)')
ax2.set_title('MA Spread (MA10 - MA100) vs Percentile Threshold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Plot 3: MA Greater condition
ax3 = axes[2]
ma_greater_series = df['MA_Greater'].astype(int)
ax3.fill_between(df.index, 0, ma_greater_series, where=(ma_greater_series == 1),
                  alpha=0.5, color='lightblue', label='MA Bullish (MA10>MA40>MA100)')
ax3.set_ylabel('Condition Met')
ax3.set_title('MA Bullish Alignment (MA10 > MA40 > MA100)')
ax3.set_ylim(-0.1, 1.1)
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['No', 'Yes'])
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

# Plot 4: Full trending condition
ax4 = axes[3]
ma_trending_series = df['MA_Trending'].astype(int)
ax4.fill_between(df.index, 0, ma_trending_series, where=(ma_trending_series == 1),
                  alpha=0.5, color='green', label='Full Trending Condition Met')
ax4.set_ylabel('Condition Met')
ax4.set_xlabel('Date')
ax4.set_title('Full MA Trending Condition (Bullish + Percentile Threshold)')
ax4.set_ylim(-0.1, 1.1)
ax4.set_yticks([0, 1])
ax4.set_yticklabels(['No', 'Yes'])
ax4.legend(loc='upper left')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('MA_trending_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n[SAVED] Visualization saved to: MA_trending_analysis.png")

# Detailed breakdown of 2023
print("\n" + "=" * 80)
print("DETAILED 2023 ANALYSIS (Month by Month)")
print("=" * 80)
df_2023 = df[df.index.year == 2023]
for month in range(1, 13):
    month_data = df_2023[df_2023.index.month == month]
    if len(month_data) > 0:
        trending_count = month_data['MA_Trending'].sum()
        trending_pct = (trending_count / len(month_data)) * 100
        avg_price = month_data['close'].mean()
        print(f"  {month:2d}/2023: Trending {trending_pct:5.1f}% ({trending_count:3d}/{len(month_data):3d} bars), "
              f"Avg Price: ${avg_price:8.2f}")

print("\n" + "=" * 80)
