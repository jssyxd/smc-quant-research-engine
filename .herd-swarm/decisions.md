# Swarm Decisions Log: Credal Uncertainty, Anti-Hallucination & Anti-Overfitting for SMC Quantitative Engine

## 2026-09-28: Inception & Architectural Synthesis

### Background
We investigated:
1. **NeurIPS 2025 Paper (arXiv:2510.12137)**: *Credal Transformer: A Principled Approach for Quantifying and Mitigating Hallucinations in Large Language Models* (Authors: Shihao Ji, Zihui Song, Jiajie Huang).
   - Core finding: Standard Softmax forces "Artificial Certainty" by collapsing ambiguous attention scores into a single point distribution.
   - Credal Solution: Replaces Softmax with Evidential Theory (Dirichlet distribution over probability simplex). When evidence is insufficient/ambiguous, the Credal set expands, enabling principled **abstention** (refusing to make a false assertion).
2. **AlphaGPT Repository (imbue-bit/AlphaGPT)**:
   - A reinforcement learning framework generating symbolic formula tokens (RPN Stack VM) for crypto meme trading.
   - Key vulnerability: As highlighted in Duan Jingyi's post, the backtest evaluation loop (`MemeBacktest`) used static in-sample cumulative returns minus penalization (`cum_ret - big_drawdowns * 2.0`), lacking Out-of-Sample verification, permutation testing, cross-asset audits, or uncertainty quantification. This led to severe **Alpha Hallucination** (overfitting to historical noise, high in-sample score but catastrophic real-world failure).

### Decisions for our SMC Quantitative Project:
1. **Decision D1: Credal Abstention in SMC Confluence Scoring**:
   - Instead of forcing an entry when Confluence Score $\ge 70$, compute Dirichlet Evidential Uncertainty across factors.
   - If Epistemic Uncertainty (ambiguity between factors or regime conflict) exceeds threshold $u_{max}$, the agent must **abstain** (`Signal = NEUTRAL`), directly mitigating false-positive trade hallucinations.
2. **Decision D2: Strict Deflationary Reward & Overfitting Metric**:
   - Deflated Sharpe Ratio (DSR) and Probability of Backtest Overfitting (PBO) will replace naive PnL metrics.
   - Any Alpha generation loop must evaluate on 80% IS and grade survival on 20% OOS.
3. **Decision D3: Disjoint Leaf Decomposition for Herdr Swarm**:
   - **Leaf A (`leaf-credal-alpha`)**: Credal evidential uncertainty engine and abstention filter for SMC signals. Owned files: `src/credal_engine.py`, `tests/test_credal.py`.
   - **Leaf B (`leaf-anti-overfit`)**: Statistical anti-overfitting suite (DSR, PBO, Combinatorial Purged Cross-Validation CPCV). Owned files: `src/anti_overfit_suite.py`, `tests/test_overfit.py`.
   - **Leaf C (`leaf-audit-report`)**: Cross-asset benchmark audit and deep-dive technical report synthesizing the Credal Transformer, AlphaGPT, and SMC integration. Owned files: `docs/research/CREDAL_ALPHAGPT_SMC_AUDIT.md`.
