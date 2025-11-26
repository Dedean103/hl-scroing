# Legacy Markdown Summary (Version 2025-11-26)

## README.md
High-level project overview for the VipHL Backtrader system: quick-start steps (venv, install, run `viphl_strategy.py`), conceptual description of the bidirectional mean-reversion logic, core features such as adaptive pivots, scoring, and risk controls, full parameter reference, toolchain pointers, and recent improvement notes. Serves as the onboarding document for developers/users.

## CLAUDE.md
Assistant-facing instructions describing core architecture (strategy class, indicators, DTOs, data feeds), required dependencies, config knobs, and development commands so that Claude-based agents know how to interact with the codebase. Essentially an internal ops guide.

## bypoint_hl_trade_examples.md
Narrative walkthrough of how byPoints, HLs, and trades relate via multiple timeline examples (single pivot, multi-pivot VVIP, trending vs normal, failed recoveries, dynamic mn impact). Emphasizes that byPoints are prerequisites, HLs aggregate them, and recovery logic dictates actual entries.

## hl_trade_scaling_implementation.md
Decision note selecting the “combined mn” approach for HL-based trade scaling. Explains why merged HLs should aggregate ByPoint mn values, how weighted averages are computed, lifecycle of mn from HL creation to trade trigger, and benefits such as signal-quality fidelity.

## hl_trade_scaling_complete_implementation.md
Comprehensive implementation report for the mn-tracking system: details code changes across HL/RecoveryWindow/TradeV2/VipHLStrategy, outlines utilities for weighted mn computation, highlights testing todo items, and enumerates example scenarios demonstrating the new behavior.

## hl_creation_dependency.md
Confirms via code references that HL objects are only created from byPoints within `rebuild_hl_from_most_recent_by_point`, showing the dependency chain byPoint → HL → recovery window → trade while noting there are no alternative creation paths.

## pnl_scaling_implementation.md
Summarizes the switch to direct position scaling (1×–3× multiplier) instead of post-trade PnL weighting. Covers the helper function, entry-size calculation, and rationale for aligning reported performance with actual exposure.

## trade_trigger_vs_bypoint_timing.md
Clarifies that trade triggers rely on recovery windows built from historical HLs, not on simultaneous byPoint detection. Provides timeline examples, describes static mn fallbacks, and reiterates that byPoints build structure while trades execute later.

## viphl-example/README.md
Minimal quick-start for the reference example repo: steps to run (`python viphl_strategy.py`), sample output metrics, and instructions for tweaking strategy parameters inside Cerebro.

## byPoint/HL/Scaling Doc Caveats
Several of the above markdown files describe intended behavior (e.g., HL mn aggregation) that may diverge from the current `viphl-source-code` snapshot; use them for historical context only and defer to `summary.md` for the latest mechanics.
