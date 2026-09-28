# Swarm Task Tree

- **Goal**: Implement Credal Uncertainty quantification (anti-hallucination) & Statistical Anti-Overfitting suite inspired by NeurIPS 2025 Credal Transformer & AlphaGPT retrospective, benchmarked against SMC multi-asset engine.
- **Captain**: Planner, reviewer, and merge queue arbiter.

## Leaves

| Leaf ID | Kind | Role | Status | Ownership | Deliverable |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **leaf-credal-alpha** | `omp` | implementer | **merged** | `src/credal_engine.py`, `tests/test_credal.py` | Credal Evidential Dirichlet Uncertainty & Abstention Filter for SMC (Passes 8 unit tests) |
| **leaf-anti-overfit** | `omp` | implementer | **merged** | `src/anti_overfit_suite.py`, `tests/test_overfit.py` | Deflated Sharpe Ratio (DSR), Probability of Backtest Overfitting (PBO), and Haircut Sharpe (Passes 6 unit tests) |
| **leaf-audit-report** | `omp` | reporter | **merged** | `docs/research/CREDAL_ALPHAGPT_SMC_AUDIT.md` | 40KB in-depth technical report & post-mortem on NeurIPS 2025 Credal Transformer & AlphaGPT |
