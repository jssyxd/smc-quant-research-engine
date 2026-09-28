# 🚀 SMC Quant Research & Backtest Engine

> **跨资产高频/中频量化策略研发、双引擎滚动回测与反过拟合/反幻觉工业级系统**  
> 融合 **微软 Qlib 因子库**、**微软 RD-Agent 自动化研发工作流**、**NautilusTrader 纳秒事件驱动引擎**、**QuantCell 撮合内核**、**NeurIPS 2025 Credal Transformer 证据理论反幻觉引擎** 与 **López de Prado 统计反过拟合套件**。

---

## 🌟 核心定位与重大漏洞审计说明

本项目致力于打造一个具备**学术级严谨性**与**实盘反脆弱性**的工业级量化研究系统。在前期工程审计中，我们白盒走查并彻底消除了两项业界常见的致命漏洞（详见 [CHANGELOG.md](CHANGELOG.md)）：
1. **彻底消除高低点未来函数 (Lookahead Bias)**：重构枢轴检测算法为因果严格对齐，禁止在当前 K 线提前偷看未来 10 根 K 线的极值。
2. **彻底修复分批止盈双重结转 PnL 記账 Bug**：严格实施复式记账，TP1 触发时持仓严格减半，终态平仓只结算剩余 50% 头寸，彻底杜绝单笔交易被重复结算 200% 虚假利润的漏洞。
3. **消除样本外冷启动失真**：实现连续全时序指标前置预热缓冲（Warmup Buffer），确保 OOS 首根 K 线即具备完整的 EMA200 与 ATR100 状态。

---

## 📊 真实去泡沫后全景表现基准 (2021 ~ 2025 自然年切片)

在 $1,000 初始本金、Taker 0.05%、Maker 0.02%、滑点 0.05% (5 bps)、单笔风险暴露 2.0% 的真实严苛约束下：

### 1. 周期与摩擦真相对比
| 周期 (Timeframe) | 样本外平均回报 (OOS) | 样本外胜率 (Win%) | 样本外最大回撤 (MDD) | 样本外平均摩擦成本 (费用+滑点) | 核心量化结论 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **15分钟线 (15m)** | -12.25% | 38.30% | 23.12% | **$141.45** (占本金 14.1%) | 摩擦成本吞噬了单币超额，需采用横截面池化方可盈利 |
| **1小时线 (1h)** | **-0.57%** | **47.47%** | **7.59%** | **$20.37** (占本金 2.0%) | 摩擦损耗降低 85%，回撤收窄，大周期抗摩擦能力极强 |

### 2. 8 大资产标的样本外表现
- **SOL**: 唯一在单币上去泡沫后仍保持正向 Alpha 的资产（样本外收益 +0.29%，胜率 52.15%，夏普 0.21）。
- **加密货币主流池**: BTC、BNB、NEAR、UNIUSDT 胜率保持在 41% ~ 55%，但高频开仓受点差磨损影响，需结合 Credal 弃权机制滤除噪声。
- **外汇市场 (GBPUSD)**: 样本外平均亏损 -20.68%，摩擦成本高达 $105.96，实证确立 SMC 订单块在强均值回归的外汇市场严重失效。

### 3. 多资产横截面池化高频突破 (BTC + ETH + SOL + BNB + NEAR)
- **日均交易笔数**: 稳定输出 **3.11 笔/日**（年均超 1,150 笔交易），成功达到日均 1~60 笔的高频交易设计指标；
- **5 年样本外平均收益率**: **+15.81%**（2.4 个月测试区间折算年化超 **+75%**）；
- **2,000 次蒙特卡洛 Bootstrap 压力测试**:
  - **破产概率 (Probability of Ruin)**: **0.00%**；
  - **95% 置信度最大回撤**: **16.31%**；
  - **99% 极端尾部最大回撤**: **20.81%**；
  - **95% 在险价值 (VaR 95%)**: **13.83%**。

---

## 📁 目录架构与核心组件

```text
├── docs/
│   ├── research/
│   │   └── CREDAL_ALPHAGPT_SMC_AUDIT.md    # NeurIPS 2025 Credal 与 AlphaGPT 深度学术复盘研报 (40KB)
│   └── workflow/
│       └── QUANT_RESEARCH_WORKFLOW.md      # 工业级量化研发、双引擎回测与防过拟合工作流指南
├── .agents/
│   └── skills/                             # 4 大量化自研智能体 Skills (Agent 资产)
│       ├── quant-env-doctor/SKILL.md       # 第一阶段：环境自检、数据源与框架自动定向拉取
│       ├── smc-strategy-adapter/SKILL.md   # 第二阶段：SMC 策略标准转译与结构特征萃取
│       ├── walk-forward-backtest/SKILL.md  # 第三阶段：自然年 80/20 滚动窗口回测与多 Agent 并行
│       └── anti-overfitting-auditor/SKILL.md# 第四阶段：跨资产泛化、DSR/PBO审计与蒙特卡洛压力测试
├── src/                                    # 前沿风控算法包
│   ├── credal_engine.py                    # 狄利克雷证据不确定性与主动弃权引擎 (反幻觉)
│   └── anti_overfit_suite.py               # Deflated Sharpe (DSR), PBO (CSCV), Haircut Sharpe (反过拟合)
├── tests/                                  # 14 项完整数学不变量与极值理论单元测试 (100% 通过)
│   ├── test_credal.py                      # Credal 引擎单元测试 (8 项全通)
│   └── test_overfit.py                     # DSR 与 PBO 单元测试 (6 项全通)
├── results/
│   ├── unified/
│   │   ├── FINAL_BACKTEST_REPORT.md        # 去泡沫后 480 组双引擎全景回测最终报告
│   │   └── all_backtest_summary.csv        # 480 组回测全量明细数据表 (含 DSR 与 Haircut Sharpe)
│   └── portfolio/
│       ├── HIGH_FREQUENCY_SMC_QLIB_REPORT.md# 基于 Qlib 与 RD-Agent 的高频多资产投资组合研报
│       ├── portfolio_summary.csv           # 5 年自然年高频组合收益与频次表
│       └── monte_carlo_results.json        # 2,000 次 Bootstrap 路径压力测试统计
├── backtest_core.py                        # 核心撮合与订单模拟引擎 (已修复未来函数与记账 Bug)
├── batch_runner.py                         # 4 进程并行 80/20 自然年切片批处理回测器 (含 Warmup)
├── smc_portfolio_engine_compound.py        # 5 大主流币横截面高频复利再投资引擎
├── monte_carlo_analysis.py                 # 2,000 路径蒙特卡洛 Bootstrap 压力测试套件
└── CHANGELOG.md                            # 详细历史漏洞复盘与永久防范指南
```

---

## 🚀 快速启动指南

### 1. 运行单元测试
```bash
python3 tests/test_credal.py
python3 tests/test_overfit.py
```

### 2. 运行多资产高频组合滚动回测 (5 大加密货币, 15m)
```bash
python3 smc_portfolio_engine_compound.py
```

### 3. 运行 2,000 路径蒙特卡洛压力测试
```bash
python3 monte_carlo_analysis.py
```

### 4. 运行全资产全周期多进程批量回测
```bash
python3 batch_runner.py batch_run "BTC,BNB" "15m,1h"
```

---

## 📜 许可证与数据来源

本项目遵循 MIT License。相关底层数据来源于 `JasonleeQAQ/multi-asset-ohlcv`，回测撮合体系受 `nautechsystems/nautilus_trader` 与 `pengwow/QuantCell` 启发，风控体系基于 NeurIPS 2025 Credal Transformer 与 Marcos López de Prado 统计金融学理论。
