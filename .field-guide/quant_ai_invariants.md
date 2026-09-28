# Quant AI Engineering Invariants (生产级量化 AI 不变量守则)

## 1. Multi-Agent Iterative Convergence Protocol (智能体多轮迭代收敛守则)
- **Hard Termination Contract (双向强制收敛)**:
  所有 Agent 针对量化策略的多轮优化与回测迭代，必须且只能收敛至以下两类最终结论之一：
  - **Conclusion 1**: 【该策略没有研究的价值，在几乎任何情况下都不具备大幅盈利的可能性、不具备作为量化交易公司的策略之一】（摩擦吞噬、排队幻觉、资金冲突下同向暴跌、样本外崩溃）。
  - **Conclusion 2**: 【该策略在严格避免过拟合的情况下能够合理优化提升 PnL】（全量标的无偏检验通过、事前物理约束单调有效、扛住严格穿透与同Bar先止损悲观检验、共享资金池复利稳健）。
- **Anti-Snooping Iron Law (反数据窥探铁律)**:
  严禁看交易记录事后挑币（Cherry-Picking）；严禁看亏损记录加特异性代码过滤；交易记录仅用于审查手续费摩擦、代码前瞻 Bug 与资金守恒。

## 2. Capital Allocation & Sizing Invariant (资金分配与头寸守则)
- **Position Sizing Formula**:
  必须反解名义资金动用比例公式：$\text{动用资金比例} = \frac{\text{Risk\%}}{\text{止损幅度 (\%) } }$。
- **Dual Capital Regimes (双资金模式检验)**:
  严禁将多标的交易次数无脑相加！必须以真实时间戳事件队列运行两版：
  1. **Isolated Capital**: 各标的独立 $1,000 USD；
  2. **Shared Capital with Opportunity Reserve**: 共享单一 $1,000 USD，限制最多 2 仓并发，永远保留 $\ge 50\%$ 现金等待其他标的机会。

## 3. Matching Realism & Pessimistic Defense (撮合真实性与悲观防爆守则)
- **Penetration Fill**: 限价单成交必须穿透 `LimitPrice ± 0.05 * ATR`，杜绝触碰即成交与排队逆向选择。
- **Intra-Bar Pessimism**: 同 Bar 触及 TP 和 SL 时强制判定先止损。
- **Dynamic Slippage**: 止损市价单滑点与 K 线实体动态放大。
