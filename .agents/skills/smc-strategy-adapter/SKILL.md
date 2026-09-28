# Skill: smc-strategy-adapter

## 用途与定位
将原始 Pine Script、TradingView 指标、SMC 策略或未对齐的交易信号转译为 **QuantCell** 和 **NautilusTrader** 读得懂的标准事件驱动/向量化策略。

## 标准转译规范与接口

### 1. 结构化因子计算输出
策略必须计算并输出以下核心 SMC 向量化特征：
1. **Pivots (高低枢轴检测)**:
   ```python
   swing_len = 6  # 自适应动态参数
   # 检测 high[i] == max(high[i-len : i+len+1])
   # 维护当前结构上的真实 HH 与 LL
   ```
2. **Order Block (订单块)**:
   ```python
   # 必须带有成交量放大确认
   vol_ratio = volume / rolling_mean(volume, 20)
   bull_ob = (close < open) & (vol_ratio >= 1.3)
   bear_ob = (close > open) & (vol_ratio >= 1.3)
   ```
3. **Fair Value Gap (FVG 公允价值缺口)**:
   ```python
   bull_fvg = (low > np.roll(high, 2)) & ((low - np.roll(high, 2)) >= 0.4 * atr)
   bear_fvg = (high < np.roll(low, 2)) & ((np.roll(low, 2) - high) >= 0.4 * atr)
   ```
4. **Liquidity Sweep (假突破扫损)**:
   ```python
   bsl_swept = (high > hh_prev) & (close < hh_prev)
   ssl_swept = (low < ll_prev) & (close > ll_prev)
   ```

### 2. 核心交易管理机制 (Trade Management)
- **非对称多空条件**: 多头 Score $\ge 72$，空头 Score $\ge 70$（加密市场多头溢价倾斜）。
- **Early Break-Even (BE) 缓冲**:
  - 浮盈达到 0.7R 止损移至：$\text{Entry} \pm 0.15 \times \text{ATR}$（多头锁微利、空头下移锁微利），防止精确保本被反抽扫损。
- **两级分批止盈**:
  - TP1 @ 1.5R: 平仓 50%（限价单 Maker 费率 0.02%）；
  - TP2 @ 3.0R: 最终平仓（限价单 Maker 费率 0.02%）。
- **止损退出**: 市价单（Taker 费率 0.05% + 滑点 0.05%）。
