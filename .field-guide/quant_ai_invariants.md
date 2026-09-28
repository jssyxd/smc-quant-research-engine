# Field Guide: Lessons, Quirks, and Invariants

## 2026-09-28: Quant Systems & AI Agent Pitfalls

### Issue: Softmax Artificial Certainty in Quant Decision-Making
- **Symptom**: Model forces high confidence $(P \approx 0.99)$ on out-of-distribution market regimes (e.g. flash crashes or liquidity voids), triggering huge drawdowns.
- **Root Cause**: Softmax normalizes exponents so relative differences force probability mass to sum to 1 regardless of absolute evidence magnitude.
- **Invariant**: Quant systems must use Evidential / Credal theory (e.g., Dirichlet concentrations $\alpha_k = e_k + 1$). Total evidence $S = \sum \alpha_k$. Epistemic uncertainty is $u = K / S$. When $u > \text{threshold}$, the system must **abstain**.

### Issue: AlphaGPT In-Sample Fitness Hallucination
- **Symptom**: `MemeBacktest` in AlphaGPT scores formulas via `score = cum_ret - big_drawdowns * 2.0`. A formula can generate 5 winning trades in a low-liquidity sample and obtain astronomical fitness, but collapses completely OOS.
- **Root Cause**: Selection bias under multiple testing without White's Reality Check or Hansen's Superior Predictive Ability test.
- **Invariant**: Never use in-sample raw returns as reinforcement learning rewards without subtracting haircut for number of trials tested (Bailey & Lopez de Prado Deflated Sharpe Ratio).
