# Swarm Decisions Log: Round 2 SMC High-Frequency Alpha Optimization & White-Hat Auditing

## 2026-09-28: Round 2 Quantitative Architectural Decisions

### Context & Diagnosis of Round 1 Post-Audit Baseline
After eliminating Lookahead Bias and the double-counting PnL bug in Round 1:
- The raw SMC strategies suffered negative returns on 15m (-12.25% OOS) primarily due to **fee & slippage friction death zone**:
  - Paying 0.05% Taker + 0.05% Slippage on entry and exit amounted to 20 bps round-trip friction.
  - On 15m, this accumulated to \$141.45 friction cost per year on a \$1,000 account (14.1% of capital).
- To achieve solid positive PnL across both In-Sample (IS) and Out-of-Sample (OOS) while operating at high frequency (1~60 trades/day), we must make fundamental quantitative upgrades:

### Decisions:
1. **Decision D4: Maker Limit Order Entry at Order Block Retracement**:
   - Instead of chasing market breakout entries (Taker 0.05% + 0.05% slippage), the strategy places Limit Orders at the 50% equilibrium of the validated Order Block / FVG.
   - Entry fee drops from 0.05% + 0.05% = 0.10% to **0.02% Maker fee with 0.00% slippage**.
   - This single change improves edge by +8 to +10 bps per trade.
2. **Decision D5: Qlib Alpha Momentum & Order Flow Imbalance (OFIP)**:
   - Confluence score incorporates Qlib-style micro-structure factors:
     - Volume Impulses ($V / \text{SMA}(V, 20) \ge 1.5$)
     - Micro price displacement ratio
     - Multi-timeframe trend alignment (15m execution conditioned on 1h EMA50)
   - Only enters when Confluence Score $\ge 70$ AND Credal Dirichlet Uncertainty $u \le 0.35$.
3. **Decision D6: Disjoint Swarm Leaf Decomposition**:
   - **Leaf A (`leaf-qlib-alpha`)**: `src/qlib_smc_alpha.py`, `tests/test_qlib_alpha.py` (Qlib feature extraction & signal generation).
   - **Leaf B (`leaf-round2-engine`)**: `src/smc_round2_engine.py`, `tests/test_round2_engine.py` (Limit order matching, compounding, OOS warmup, and metric pipeline).
   - **Leaf C (`leaf-whitehat-auditor`)**: `tests/test_whitehat_security.py` (White-hat penetration testing, mathematical invariants, lookahead proofs, balance conservation).
