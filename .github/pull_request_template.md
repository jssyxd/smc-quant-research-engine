# GitHub Pull Request Template: Agent Research Deliverable
# ==============================================================================
# Mandatory Checklist for ANY Automated or Human PR in this Quant Engine Repo
# ==============================================================================

## 🎯 Dual-Convergence Protocol Verification (Mandatory Gate)

> **WARNING**: Any PR will be automatically rejected by GitHub Actions CI if it fails
> to terminate definitively at exactly ONE of the two allowed conclusions.

### Final Research Verdict (Check exactly ONE):
- [ ] **【终局结论 1：该策略没有研究的价值，在几乎任何情况下都不具备大幅盈利的可能性、不具备作为量化交易公司的策略之一】**
  - *Action*: Strategy marked `[STATUS: ABANDONED]`, falsification report filed in `docs/research/`, no live deployment.
- [ ] **【终局结论 2：该策略在严格避免过拟合的情况下能够合理优化提升 PnL】**
  - *Action*: Strategy marked `[STATUS: PRODUCTION_READY]`, passed 100% whitehat tests, production code delivered.

---

## 📋 Quantitative Invariant Audit Checklist

Before requesting review or merging, verify that the agent followed all mandatory stages:

- [ ] **1. Position Sizing Formula Audited**: Formula defined as $\text{Notional} = \text{Equity} \times \frac{\text{Risk\%}}{\text{StopDistance\%}}$ with $\le 4.0x$ leverage cap.
- [ ] **2. Zero Cherry-Picking Verified**: All applicable sovereign crypto assets (BTC, ETH, BNB, SOL) tested without post-hoc removal of losing symbols.
- [ ] **3. Dual Capital Regimes Tested**:
  - [ ] Tested Isolated Capital Pool ($1,000 / asset independent).
  - [ ] Tested Shared Capital Pool ($1,000 total, max 2 concurrent positions, $\ge 50\%$ cash reserve).
- [ ] **4. Realistic Pessimistic Matching Audited**:
  - [ ] Strict penetration fill (`Low < Limit - 0.05*ATR`).
  - [ ] Intra-bar TP/SL touches strictly executed as Stop-Loss first.
  - [ ] Dynamic ATR slippage applied.
- [ ] **5. CI Verification Passed**:
  - [ ] `make test` executed cleanly (100% pass).
  - [ ] `scripts/validate_agent_convergence.py` validated the deliverable.
