# Swarm Task Tree: Round 2 SMC High-Frequency Alpha Optimization & White-Hat Auditing

- **Goal**: Boost IS/OOS PnL while maintaining 1~60 trades/day via Qlib micro-structure factors, Maker limit entries, compounding, 2021-2025 walk-forward backtesting, 2000-path Monte Carlo stress tests, and automated white-hat auditing.
- **Captain**: Planner, reviewer, and merge queue arbiter.

## Leaves

| Leaf ID | Kind | Role | Status | Ownership | Deliverable |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **leaf-qlib-alpha** | `omp` | implementer | planned | `src/qlib_smc_alpha.py`, `tests/test_qlib_alpha.py` | Qlib Micro-Structure Alpha Pipeline (OFIP, Volume Delta, OB 50% Entry Zone) |
| **leaf-round2-engine** | `omp` | implementer | planned | `src/smc_round2_engine.py`, `tests/test_round2_engine.py` | Execution Engine with Maker Limit Entries, Compounding, Credal Abstention, and Walk-Forward Pipeline |
| **leaf-whitehat-auditor** | `omp` | reviewer | planned | `tests/test_whitehat_security.py` | Automated White-Hat Security, Invariant Verification, and Lookahead Zero-Proof |
