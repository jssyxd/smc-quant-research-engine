# Skill: walk-forward-backtest

## 用途与定位
工业级滚动窗口回测与多 Agent 并行执行引擎。严格遵循自然年切片、80% 样本内 (IS) 训练 + 20% 样本外 (OOS) 盲测、全量摩擦成本与复利再投资计算。

## 标准执行流程

### 1. 自然年切片与严格防前视偏差划分
对于回测区间内的每一个自然年（例如 2021 ~ 2025）：
```python
year_df = full_df[full_df.index.year == year]
split_idx = int(len(year_df) * 0.8)
df_in_sample = year_df.iloc[:split_idx]       # 前 80% 作为样本内
df_out_of_sample = year_df.iloc[split_idx:]   # 后 20% 作为样本外
```
- **样本内**: 用于参数适应、因子权重确立；
- **样本外**: 参数绝对冻结，进行盲测前向检验。

### 2. 精确成本与资金模型
- **初始账户本金**: `$1,000.00 USD`；
- **手续费**:
  - 市价入场与止损单：**Taker 0.05% (-0.0005)**；
  - 挂单止盈单（TP1 / TP2）：**Maker 0.02% (-0.0002)**。
- **滑点冲击**: **0.05% (5 bps)**（买入价格上浮 1.0005，卖出价格下浮 0.9995）；
- **动态风险仓位管理**:
  $$\text{Position Size} = \frac{\text{Current Equity} \times 2\%}{\text{Stop Distance}}$$
- **复利开关**: 支持净值动态复利再投资（`use_compound = True`）或固定本金模式。

### 3. 多 Agent 并行加速与内存隔离
- 采用进程级工作池将不同标的（如 BTC/BNB, SOL/UNI, NEAR/XAU, USO/GBP）分配给独立 Agent/Worker；
- 单进程完成计算后强制垃圾回收（`gc.collect()`），防止大时间跨度高频数据累积引发 OOM 异常。
