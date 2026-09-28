# 📋 项目更新日志与重大缺陷复盘 (CHANGELOG & PITFALL POST-MORTEM)

本文档记录本项目从初始回测、白盒代码审计、漏洞捕获、数学与会计层修复，到最终引入前沿反幻觉（Credal Transformer）与反过拟合（López de Prado DSR）体系的完整演进全貌。**重点复盘之前踩过的重大陷阱与漏洞，为后续研究与实盘部署提供永久警示。**

---

## [v2.1.0] - 2026-09-28 · 深度代码审计、漏洞消除与真实基线确立

### ⚠️ 重大踩坑复盘与修复要点 (Critical Pitfalls & Bug Fixes)

#### 1. 致命缺陷一：结构高低点（Swing High/Low）存在未来函数 (Lookahead Bias)
- **踩坑现象**:
  在 `indicators.py` 与早期微观结构引擎中，枢轴极值点检测采用了对称切片：
  ```python
  # ❌ 错误代码 (包含向前看未来数据的严重未来函数)：
  for i in range(swing_len, n - swing_len):
      if high[i] == np.max(high[i - swing_len : i + swing_len + 1]):
          is_high[i] = True
  for i in range(n):
      if is_high[i]:
          hh[i] = high[i] # 👈 在第 i 根 K 线就赋予了高点
  ```
- **机理危害**:
  在真实的实盘和逐 bar 撮合中，要确认第 $i$ 根 K 线是这 $2 \times swing\_len + 1$ 根 K 线中的最高点，必须等待未来的第 $i+1 \sim i+swing\_len$ 根 K 线全部走完后方可确立。原代码直接在第 $i$ 根 K 线赋予 `hh[i] = high[i]`，导致后续的 `bull_bos = close > np.roll(hh, 1)` 与假突破扫损 `bsl_swept` 在第 $i+1$ 根 K 线直接偷看了未来 10 根 K 线的局部极值，大幅虚增了历史回测胜率与收益。
- **永久修复方案 (Causal Pivot Confirmation)**:
  全面重构为因果严格对齐的时间轴检测：只有在当前时间 $t$ 满足时，才将 $t - swing\_len$ 处的极值确认为历史高/低点，即对于所有指标访问，绝对只使用 $\le t$ 的信息：
  ```python
  # ✅ 修复后代码 (100% 因果严格对齐，零未来函数)：
  for t in range(2 * swing_len, n):
      p = t - swing_len
      if high[p] == np.max(high[p - swing_len : t + 1]):
          curr_hh = high[p]
      if low[p] == np.min(low[p - swing_len : t + 1]):
          curr_ll = low[p]
      hh[t] = curr_hh
      ll[t] = curr_ll
  ```

---

#### 2. 致命缺陷二：分批止盈 (TP1/TP2) 双重结转 PnL 记账 Bug (Double-Counting Accounting Bug)
- **踩坑现象**:
  在 `backtest_core.py` 与 `smc_strategy.py` 的撮合逻辑中：
  ```python
  # ❌ 错误代码 (TP1 结转了全额利润却未减仓)：
  if position == 1 and high >= tp1:
      fee = pos_size * exit_px * self.maker_fee
      pnl = pos_size * (exit_px - entry_price) - fee # 👈 按 100% 仓位算了全额利润并加到 cash
      cash += pnl
      tp1_hit = True
      # ⚠️ 遗漏了 pos_size = pos_size * 0.5 的减半扣减！
  ...
  # 随后打到 TP2 或止损时：
  if position == 1 and high >= tp2:
      fee = pos_size * exit_px * self.maker_fee
      pnl = pos_size * (exit_px - entry_price) - fee # 👈 此时 pos_size 依然是 100%，再次结算全额利润！
      cash += pnl
  ```
- **机理危害**:
  一旦单笔交易打中 TP1 并随后到达 TP2 或被打止损，资金池 `cash` 会被重复计入利润，导致账面产生单笔高达 200% 的虚假利润结转，严重扭曲了盈亏比和资金曲线。
- **永久修复方案 (Double-Entry Financial Accounting)**:
  实施复式记账原则，TP1 触发时严格减仓 50%：
  ```python
  # ✅ 修复后代码 (严格守恒记账)：
  close_size = initial_pos_size * 0.50
  fee = close_size * exit_px * self.maker_fee
  pnl_part = close_size * (exit_px - entry_price) - fee
  cash += pnl_part
  pos_size = initial_pos_size - close_size  # 剩余 50% 仓位
  realized_tp1_pnl = pnl_part
  # 后续终态平仓时，仅对剩余的 pos_size 结算，单笔交易总利润严格等于 realized_tp1_pnl + pnl_rem
  ```

---

#### 3. 陷阱三：样本外 (OOS) 冷启动无预热失真 (Cold-Start Bias)
- **踩坑现象**:
  在做 80% IS / 20% OOS 切片时，直接从 OOS 的第一根 K 线截断输入指标计算函数，导致 `ATR100`、`EMA200` 等需要较长周期的指标在 OOS 开头数十到上百根 K 线内处于缺失或未收敛状态，产生虚假信号。
- **永久修复方案 (Continuous Warmup Buffer)**:
  在切片前，利用全量连续历史时序完成 EMA、ATR 及 Swing 极值的因果预热，切片时直接提取包含完整预热状态的特征矩阵（`data_is` 和 `data_oos`），完全消除冷启动失真。

---

#### 4. 陷阱四：单币高频交易的“摩擦死区”幻觉 (Friction Death Zone)
- **踩坑现象**:
  试图通过在单币（如 BTC 5m）上过度放宽入场阈值来刷频到日均 10~20 笔。
- **机理危害**:
  在每笔 0.05% Taker + 0.05% 滑点（单次回合摩擦达 0.1% ~ 0.2%）的现实约束下，高频次、低胜率的伪信号会导致交易成本迅速吃光所有 Alpha，年摩擦成本突破 $4,000（远超初始本金 $1,000）。
- **永久修复方案 (Cross-Asset Pooling)**:
  放弃单币刷频，采用**多资产横截面池化架构**（BTC, ETH, SOL, BNB, NEAR 组合），每个币种维持严格高胜率（日均 0.5~0.7 笔），组合整体自然达到日均 2.8~3.4 笔的高频交易目标，同时将交易摩擦牢牢控制在可承受区间。

---

#### 5. 陷阱五：AlphaGPT 式“回测拟合强化学习”导致的 Alpha 幻觉 (Alpha Hallucination)
- **复盘反思**:
  AlphaGPT（`imbue-bit/AlphaGPT`）的 `MemeBacktest` 仅以单一历史样本内的简单指标作为强化学习奖励，导致模型过度拟合历史噪声，在实盘遭遇滑点和未见行情时崩溃。
- **永久修复方案**:
  引入 **Credal 狄利克雷证据不确定性拒识断路器 (`src/credal_engine.py`)** 与 **Deflated Sharpe Ratio (DSR) 多重试验方差折价 (`src/anti_overfit_suite.py`)**。当市场出现证据冲突或噪音过大时，系统拥有主动“弃权不交易（Abstain）”的能力，从根本上杜绝盲目开仓的幻觉。

---

## 🚀 后续开发与回测铁律 (Future Commandments)
1. **零容忍未来函数**: 所有价格衍生因子（HH/LL/Pivot/ZigZag）必须严格保证在时刻 $t$ 只使用 $\le t$ 的已结束 K 线；
2. **复式记账校验**: 凡涉及分批止盈/减仓逻辑，必须满足 `持仓数量 + 已平数量 = 初始开仓量` 的守恒约束；
3. **样本外必须预热**: 所有滑动窗口、Walk-Forward 回测必须携带至少 250 根 K 线的指标预热数据；
4. **全量计提摩擦成本**: 任何高频回测必须严格模拟 0.05% Taker / 0.02% Maker 费率结构并附加 5 bps 滑点。
