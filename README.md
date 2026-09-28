# 🔬 SMC Quant Research Engine & Multi-Agent Backtest Swarm

生产级智能体协作量化研发、跨资产高频/中频 SMC 策略评测、双引擎撮合审计与反过拟合工业级引擎。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NautilusTrader: Compatible](https://img.shields.io/badge/NautilusTrader-1.227%2B-green.svg)](https://github.com/nautechsystems/nautilus_trader)
[![QuantCell: Supported](https://img.shields.io/badge/QuantCell-Integrated-orange.svg)](https://github.com/pengwow/QuantCell)

---

## 🎯 智能体核心工作流收敛协议 (Dual Convergence Protocol)

本项目严格遵循 [`docs/workflow/QUANT_RESEARCH_WORKFLOW.md`](docs/workflow/QUANT_RESEARCH_WORKFLOW.md) 工业级量化工作流规范。所有多 Agent 蜂群协作的策略优化与迭代，**严禁无限期调参，严禁事后挑选优势品种（Cherry-Picking），最终必须且只能收敛至以下两种确定性结论之一**：

1. **【终局结论 1：该策略没有研究的价值】**  
   *在几乎任何情况下都不具备大幅盈利的可能性、不具备作为量化交易公司的策略之一。*  
   - 扣除真实 Maker/Taker 手续费与滑点后期望为负；
   - 依赖“挂单触碰即成交”排队幻觉或“同 Bar 优先止盈”虚假红利；
   - 在多标的共享资金池（限 2 仓并发 + 留存 50% 现金）下同向暴跌共振；
   - 样本外盲测崩溃。
2. **【终局结论 2：该策略在严格避免过拟合的情况下能够合理优化提升 PnL】**  
   *全量适用标的无偏检验通过，事前理论约束有效减亏增益，具备作为量化公司实盘配置子策略的价值。*  
   - 零事后挑选，BTC/ETH/BNB/SOL 主权加密池全量检验通过；
   - 事前（Ex-Ante）Credal 狄利克雷认知不确定性约束在盲测中呈现严格单调的减亏/增益效果；
   - 扛住“严格穿透成交 + 同 Bar 优先止损”的极度悲观撮合考验；
   - 共享资金池实序模拟下具备稳健复利能力。

---

## 📌 核心标的与资金分配模型审计

针对策略作者声明适用的四大主流加密资产（**BTC、ETH、BNB、SOL**），进行了 2021~2025 年自然年滚动窗口（80% IS / 20% OOS）下的独立资金池 vs 共享资金池深度对照：

### 1. 策略每次交易动用多少比例账户资金？
源码（`smc_strategy.py:122-131`）并非固定百分比买入，而是按**固定风险比例（2.0%）结合止损点差反算名义资金**：
$$\text{动用资金比例} = \frac{2.0\%}{\text{止损幅度 (\%) }}$$
- **止损 1% 时**：动用 200% 资金（开 2 倍杠杆）；
- **止损 2% 时**：刚好动用 100% 账户资金；
- **止损 4% 时**：动用 50% 资金（留存 50% 现金）；
- **止损大于 5% 时**：代码强制弃单，设立 4.0x 最大名义杠杆安全红线。

### 2. 双资金模型回测对比 (1h 周期 5 年自然年盲测 OOS)
- **版本 A：不共用资金（每个标的独立 $1,000）**
  - **SOL**: +0.70%/年 (胜率 63.6%, 回撤 3.17%)
  - **BTC**: -0.97%/年 (胜率 37.2%, 回撤 3.62%)
  - **ETH**: -1.14%/年 (胜率 46.8%, 回撤 4.98%)
  - **BNB**: -1.82%/年 (胜率 39.6%, 回撤 5.41%)
- **版本 B：共用资金池（$1,000 总本金，限最多 2 仓并发，保留 50% 资金等待机会）**
  - 年均实际执行：**27.2 笔/年**；
  - 因满仓错失机会：**仅 0.6 笔/年（错失率 2.2%）**，说明由于 SMC 信号天然稀疏，绝大多数时间资金充沛；
  - 样本外年均收益：**-2.71%**，平均胜率：**45.3%**，平均最大回撤：**8.8%**；
  - 最大风险：四大币种高度正相关，2024 年下半年双仓同时被打穿止损造成单年回撤 12.51%。

---

## 🛠️ 核心架构与工程交付物

1. **生产级极度悲观撮合执行引擎** (`src/smc_pessimistic_engine.py`):
   - **严格盘口穿透机制 (`Strict Penetration Fill`)**: 挂单成交必须满足价格穿透 `LimitPrice ± 0.05 * ATR`，彻底消除时间优先队列下的排队幻觉与逆向选择；
   - **同 Bar 极端悲观裁决 (`Pessimistic Intra-Bar Conflict`)**: 单根 K 线同时满足止盈与止损时，强制裁决为先扫损出局，杜绝假保本红利；
   - **动态恐慌滑点 (`Dynamic Panic Slippage`)**: 止损单滑点与 K 线实体波幅联动放大。
2. **多资产时序事件队列执行器** (`crypto_capital_models_runner.py`):
   - 支持独立资金池与共享资金池并发控制（最多 2 仓并发 + 留存 50% 机会资金）。
3. **白帽安全性与数学不变性测试** (`tests/test_whitehat_security.py`):
   - 100% 通过未来时序打乱不变性与复式记账平衡守恒测试。
4. **研报体系**:
   - 工作流最高宪法: [`docs/workflow/QUANT_RESEARCH_WORKFLOW.md`](docs/workflow/QUANT_RESEARCH_WORKFLOW.md)
   - 资金分配模型深度审计: [`docs/research/CRYPTO_CAPITAL_ALLOCATION_AUDIT.md`](docs/research/CRYPTO_CAPITAL_ALLOCATION_AUDIT.md)
   - 悲观撮合全量对比研报: [`docs/research/PESSIMISTIC_MATCHING_REPORT.md`](docs/research/PESSIMISTIC_MATCHING_REPORT.md)
   - 反过拟合形式化审计底稿: [`docs/research/ANTI_OVERFITTING_CREDAL_AUDIT.md`](docs/research/ANTI_OVERFITTING_CREDAL_AUDIT.md)
