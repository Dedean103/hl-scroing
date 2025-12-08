## Implementation Tracker

### Current progress
- **Dynamic m/n params** (Done): both strategy entry points expose `mn_start_point_*` / `mn_cap_*` knobs and pass them through `Settings`.
- **Dynamic pivot detector** (Done): `PivotHigh`/`PivotLow` now keep “live” candidates, expand `(m,n)` until failure/cap, include boundary safeguards, and output the final `(m,n)` via extra lines.
- **ByPoint plumbing** (Done): `VipHL.add_new_by_points_to_pending` reads the new `n/m` lines, validates trend alignment, and injects the actual values into `ByPoint`s so HL logic works with real windows.
- **Docs/plan** (Done): `dynamic_mn_implementation_plan.md` has steps 1‑3 marked complete; `discussion_summary.md` reflects the new behaviour.
- **Visualization runner** (Done, committed 2025‑12‑01): `hl-scroing/run_viphl_and_plot.py` consolidates the old plotting helpers into a CLI that runs VipHL with the dynamic `(m,n)` logic, prints the Backtrader stats, and emits the refreshed charts/stats without touching the stale `plotting/` module.
- **Debug log path** (Done): `debug_mode` plus `debug_log_path` now streams every ByPoint/HL event to a Markdown file so you can review construction history after a run.

### Next steps
1. **Testing + verification**: add regression scripts to ensure pivot upgrades behave as expected, re-run the Backtrader strategies to confirm the new sizing/logging doesn’t regress behaviour, and capture a sample Markdown trace for reference.
2. **Docs/Examples**: update README or a dedicated design note showing how to tune `mn_start_point_*`/`mn_cap_*`, how weighted `(m,n)` affect sizing, and how to enable/inspect the Markdown debug log (CLI default, sample output).
