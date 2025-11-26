# Dynamic mn / HL integration summary (2025-11-26)

## What was done
- Replaced the naive on-the-fly dynamic mn scanning with a candidate queue inside `VipHL`. Each pivot that fires now enters `pending_dynamic_by_points`, grows its window only once the required right-side bars are available, and only finalizes when either the next window fails or the configured `max_mn_cap` is reached.  
- Extended the DTOs: `ByPoint`/`HL` now carry the chosen `pivot_m/n`, `HL.merge` averages them, and recovery windows propagate `avg_pivot_m/n` through `from_recovery_window_result_v2`.  
- Strategy plumbing now feeds the stored `(m,n)` from the recovering HL into `quote_trade`/`record_trade`; the previous `find_dynamic_pivot` lookup remains as a fallback but is no longer used for the actual scaling when a recovery signal exists.
- Added `pending_dynamic_by_points` initialization everywhere `VipHL` is instantiated so the new queue is available in every strategy variant.

## How the dynamic mn works now
- Each pivot signal (normal/trending high/low) creates a `DynamicByPointCandidate` with `next_window` initialized to `dynamic_mn_start`.  
- On each new bar `process_dynamic_candidates` iterates the pending list; for each candidate it checks whether enough bars exist to test the next window and validates the pivot across `[pivot - window, pivot + window]`.  
- If the validation passes, the candidate advances `next_window` (by `dynamic_mn_step`) and records the last successful window; the loop stops only when reaching `max_mn_cap` or when validation fails.  
- Once a candidate finalizes, we append a `ByPoint` that carries the winning `(m,n)` into HL construction, ensuring that the recovered HL and every downstream trade see the precise window that the swing actually satisfied.

## Remaining work / sanity checks
1. Run the usual backtests/debug logging to verify pivots climb (4→5→…) as bars arrive and that trades log the HL-provided `(m,n)` during entries/exits.  
2. Validate that merged HLs still behave sensibly when their averaged `avg_pivot_m/n` is used for sizing—if needed, consider alternative merge policies (e.g., keep the latest pivot’s window instead of averaging).  
3. Update any visualization tooling (`trade_visualization.py`, etc.) to expose the new HL window metadata on the charts for easier inspection.

## Tests
- Not run (not requested); recommend full backtest with debug logging and compare logging output before/after to confirm scaling changes.
