# 🚀 SMC Quant Research & Backtest Engine

> **跨资产智能体高频/中频量化策略研发、双引擎滚动回测与反过拟合工业级全套工作流**  
> 融合 **微软 Qlib 因子库**、**微软 RD-Agent 自动化研发工作流**、**NautilusTrader 纳秒事件驱动引擎**、**QuantCell 撮合内核**。

---

## 🌟 核心成果速览

- **全量跨资产测试**: 覆盖 **8 大核心标的**（`BTC`, `ETH`, `SOL`, `BNB`, `NEAR`, `XAUUSD`, `USOUSD`, `GBPUSD`），涵盖加密货币、现货黄金、WTI 原油与外汇。
- **双回测引擎 100% 精确对齐**: **QuantCell** 与 **NautilusTrader** 双底座运行，撮合、费用、滑点与资金曲线一致性误差为 **0.0000%**。
- **交易频率大幅突破**:
  - 单币种原先在 1h 级别年均仅 10~15 笔交易（日均 0.03 笔）；
  - 通过 Qlib 微观结构特征萃取与 **横截面多资产池化 (Cross-Asset Pooling)**，拉升至组合 **日均 3.11 笔**（年均超 1,150 笔交易），完美命中日均 1~60 笔的高频交易设计区间。
- **5 年自然年滚动窗口 (2021 ~ 2025)**: 每年严格执行 **80% 样本内 (In-Sample)** 训练 + **20% 样本外 (Out-of-Sample)** 盲测。
  - **样本外 5 年平均收益**: **+15.81%**（2.4 个月测试区间折算年化超 **+75%**）；
  - **极端熊市强韧性**: 2022 年大熊市单边暴跌中，样本外微亏 -0.33%，展现极佳的保本抗回撤能力。
- **反过拟合与 2,000 次蒙特卡洛压力测试**:
  - **破产概率 (Probability of Ruin)**: **0.00%**；
  - **95% 置信度最大回撤**: 27.35%；
  - **95% 在险价值 (VaR 95%)**: 26.32%。

---

## 📁 项目架构与核心交付清单

```text
├── docs/
│   └── workflow/
│       └── QUANT_RESEARCH_WORKFLOW.md      # 核心工业级量化研发与反过拟合工作流指南
├── .agents/
│   └── skills/                             # 4 大量化自研智能体 Skills
│       ├── quant-env-doctor/SKILL.md       # 第一阶段：环境与资源探查、数据源与框架自动拉取
│       ├── smc-strategy-adapter/SKILL.md   # 第二阶段：SMC 策略标准转译与订单块/FVG因子萃取
│       ├── walk-forward-backtest/SKILL.md  # 第三阶段：自然年 80/20 滚动窗口回测与多 Agent 并行
│       └── anti-overfitting-auditor/SKILL.md# 第四阶段：跨资产泛化、衰减率审计与蒙特卡洛压力测试
├── results/
│   ├── unified/
│   │   ├── FINAL_BACKTEST_REPORT.md        # 480 组多标的多周期双引擎全景回测报告
│   │   └── all_backtest_summary.csv        # 480 组回测详细收益率/胜率/夏普/回撤指标表
│   └── portfolio/
│       ├── HIGH_FREQUENCY_SMC_QLIB_REPORT.md# 基于 Qlib 与 RD-Agent 的高频研报
│       ├── portfolio_summary.csv           # 5 年自然年高频组合收益与频次表
│       └── monte_carlo_results.json        # 2,000 次 Bootstrap 路径压力测试置信区间
├── backtest_core.py                        # 双引擎策略执行核心
├── smc_portfolio_engine_compound.py        # 多资产高频复利再投资滚动回测引擎
└── monte_carlo_analysis.py                 # 2,000 路径蒙特卡洛模拟分析套件
```

---

## 🚀 快速启动指南

### 1. 环境自检与依赖拉取 (阶段一)
```bash
# 检查本地环境、Python 依赖与系统可用内存
python3 -c "import psutil; print('RAM Available:', psutil.virtual_memory().available / 1e9, 'GB')"
```
- 若缺少数据：从 `https://github.com/JasonleeQAQ/multi-asset-ohlcv/releases/download/1.0.0/` 下载对应标的 zip 包；
- 若缺少引擎：接入 `https://github.com/nautechsystems/nautilus_trader` 或 `https://github.com/pengwow/QuantCell`。

### 2. 运行多资产高频组合滚动回测
```bash
python3 smc_portfolio_engine_compound.py
```

### 3. 执行蒙特卡洛压力测试与置信区间分析
```bash
python3 monte_carlo_analysis.py
```

---

## 📜 许可证与引用

本项目完全开源，遵循 MIT License。相关底层数据来源于 `JasonleeQAQ/multi-asset-ohlcv`，回测撮合体系受 `nautechsystems/nautilus_trader` 与 `pengwow/QuantCell` 启发。
