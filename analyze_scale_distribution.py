"""
Analyze the distribution of pnl_scale between winning and losing trades
for k values 1.1, 1.2, and 1.3 with m=n=4
"""
import backtrader as bt
from viphl_strategy_scoring import VipHLStrategy, load_data_from_csv

class ScaleAnalysisStrategy(VipHLStrategy):
    """Extended strategy to capture scale data"""

    def __init__(self):
        super().__init__()
        self.combined_scores = []  # Track all combined scores

    def record_trade(self, extend_bar_signal_offset):
        """Override to capture combined_score before creating trade"""
        # Calculate HL byP scores for current market condition
        if self.is_ma_trending[0]:
            high_score = self.calculate_hl_byp_score(
                self.p.high_by_point_m_on_trend,
                self.p.high_by_point_n_on_trend,
                pivot_type='high',
                is_trending=True
            )
            low_score = self.calculate_hl_byp_score(
                self.p.low_by_point_m_on_trend,
                self.p.low_by_point_n_on_trend,
                pivot_type='low',
                is_trending=True
            )
        else:
            high_score = self.calculate_hl_byp_score(
                self.p.high_by_point_m,
                self.p.high_by_point_n,
                pivot_type='high',
                is_trending=False
            )
            low_score = self.calculate_hl_byp_score(
                self.p.low_by_point_m,
                self.p.low_by_point_n,
                pivot_type='low',
                is_trending=False
            )

        # Combined score for trade size adjustment with scaling factors
        weighted_high = high_score * self.p.high_score_scaling_factor
        weighted_low = low_score * self.p.low_score_scaling_factor
        total_weight = self.p.high_score_scaling_factor + self.p.low_score_scaling_factor
        combined_score = (weighted_high + weighted_low) / total_weight

        # Store combined score
        self.combined_scores.append(combined_score)

        # Call parent implementation
        super().record_trade(extend_bar_signal_offset)

    def stop(self):
        """Override stop to analyze scale distribution before displaying results"""
        # Collect scale data
        winner_scales = []
        loser_scales = []
        winner_pnls = []
        loser_pnls = []
        all_scales = []

        for trade in self.trade_list:
            scale = self.trade_scales.get(id(trade), 1.0)
            all_scales.append(scale)

            if trade.pnl > 0:
                winner_scales.append(scale)
                winner_pnls.append(trade.pnl)
            else:
                loser_scales.append(scale)
                loser_pnls.append(trade.pnl)

        # Check for scale variance
        unique_scales = set(all_scales)
        unique_combined_scores = set([round(s, 6) for s in self.combined_scores])

        print(f"\nTotal trades: {len(all_scales)}")
        print(f"Unique combined_scores: {len(unique_combined_scores)}")
        if len(unique_combined_scores) <= 5:
            print(f"Combined score values: {sorted(unique_combined_scores)}")
        print(f"Unique scales: {len(unique_scales)}")
        if len(unique_scales) <= 5:
            print(f"Scale values: {sorted(unique_scales)}")

        # Print analysis
        print("\n" + "="*70)
        print(f"PNL SCALE DISTRIBUTION ANALYSIS (k={self.p.power_scaling_factor})")
        print("="*70)

        if winner_scales:
            print(f"\n{'WINNING TRADES':^70}")
            print("-"*70)
            print(f"  Count:        {len(winner_scales)}")
            print(f"  Avg Scale:    {sum(winner_scales)/len(winner_scales):.3f}")
            print(f"  Min Scale:    {min(winner_scales):.3f}")
            print(f"  Max Scale:    {max(winner_scales):.3f}")
            print(f"  Total PnL:    {sum(winner_pnls):.2f}%")

            # Calculate unscaled PnL
            unscaled_win_pnl = sum(pnl/scale for pnl, scale in zip(winner_pnls, winner_scales))
            print(f"  Unscaled PnL: {unscaled_win_pnl:.2f}%")
            print(f"  Scale Impact: {sum(winner_pnls) - unscaled_win_pnl:+.2f}%")

        if loser_scales:
            print(f"\n{'LOSING TRADES':^70}")
            print("-"*70)
            print(f"  Count:        {len(loser_scales)}")
            print(f"  Avg Scale:    {sum(loser_scales)/len(loser_scales):.3f}")
            print(f"  Min Scale:    {min(loser_scales):.3f}")
            print(f"  Max Scale:    {max(loser_scales):.3f}")
            print(f"  Total PnL:    {sum(loser_pnls):.2f}%")

            # Calculate unscaled PnL
            unscaled_loss_pnl = sum(pnl/scale for pnl, scale in zip(loser_pnls, loser_scales))
            print(f"  Unscaled PnL: {unscaled_loss_pnl:.2f}%")
            print(f"  Scale Impact: {sum(loser_pnls) - unscaled_loss_pnl:+.2f}%")

        # Calculate fit scores
        if winner_pnls and loser_pnls:
            scaled_win_total = sum(winner_pnls)
            scaled_loss_total = sum(loser_pnls)
            unscaled_win_total = sum(pnl/scale for pnl, scale in zip(winner_pnls, winner_scales))
            unscaled_loss_total = sum(pnl/scale for pnl, scale in zip(loser_pnls, loser_scales))

            scaled_fit = -scaled_win_total / scaled_loss_total if scaled_loss_total != 0 else float('inf')
            unscaled_fit = -unscaled_win_total / unscaled_loss_total if unscaled_loss_total != 0 else float('inf')

            print(f"\n{'FIT SCORE COMPARISON':^70}")
            print("-"*70)
            print(f"  Scaled Fit Score:     {scaled_fit:.2f}")
            print(f"  Unscaled Fit Score:   {unscaled_fit:.2f}")
            print(f"  Difference:           {scaled_fit - unscaled_fit:+.2f}")
            if unscaled_fit != 0:
                print(f"  Impact:               {((scaled_fit - unscaled_fit) / unscaled_fit * 100):+.1f}%")

        # Compare average scales
        if winner_scales and loser_scales:
            avg_winner_scale = sum(winner_scales) / len(winner_scales)
            avg_loser_scale = sum(loser_scales) / len(loser_scales)

            print(f"\n{'SCALE COMPARISON':^70}")
            print("-"*70)
            print(f"  Winners avg scale:  {avg_winner_scale:.3f}")
            print(f"  Losers avg scale:   {avg_loser_scale:.3f}")
            print(f"  Difference:         {avg_winner_scale - avg_loser_scale:+.3f}")

            if avg_winner_scale > avg_loser_scale:
                print(f"\n  ✓ Winners have HIGHER average scale ({(avg_winner_scale/avg_loser_scale - 1)*100:.1f}% higher)")
                print(f"    This AMPLIFIES the fit score positively!")
            elif avg_winner_scale < avg_loser_scale:
                print(f"\n  ✗ Winners have LOWER average scale ({(1 - avg_winner_scale/avg_loser_scale)*100:.1f}% lower)")
                print(f"    This REDUCES the fit score!")
            else:
                print(f"\n  = Winners and losers have EQUAL average scales")
                print(f"    Scale has NO NET EFFECT on fit score ratio")

        print("="*70 + "\n")

        # Don't call parent's stop to avoid duplicate output


def run_analysis_for_k(k_value, csv_file='BTC.csv'):
    """Run analysis for a specific k value"""
    cerebro = bt.Cerebro()

    # Add data
    dataframe = load_data_from_csv(csv_file)
    pandasData = bt.feeds.PandasData(dataname=dataframe)
    cerebro.adddata(pandasData)
    cerebro.broker.set_coc(True)

    # Add strategy with analysis
    cerebro.addstrategy(
        ScaleAnalysisStrategy,
        high_by_point_n=4,
        high_by_point_m=4,
        low_by_point_n=4,
        low_by_point_m=4,
        high_by_point_n_on_trend=4,
        high_by_point_m_on_trend=4,
        low_by_point_n_on_trend=4,
        low_by_point_m_on_trend=4,
        power_scaling_factor=k_value,
        mintick=0.01,
        close_above_low_threshold=0.5,
        debug_mode=False
    )

    cerebro.run()


if __name__ == '__main__':
    print("="*70)
    print("SCALE DISTRIBUTION ANALYSIS FOR DIFFERENT K VALUES")
    print("Configuration: m=n=4 for all parameters")
    print("="*70)

    for k in [1.1, 1.2, 1.3]:
        run_analysis_for_k(k)
        print("\n")
