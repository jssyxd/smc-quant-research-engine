# Skill: anti-overfitting-auditor

## 用途与定位
量化策略防过拟合审查、泛化能力审计与极端黑天鹅压力测试工具包。包含跨资产横截面检验、样本内外衰减比审计以及 2000 路径蒙特卡洛 Bootstrap 模拟。

## 防过拟合三层审计体系

### 1. 跨资产横截面普适性审计 (Cross-Asset Audit)
- 严禁策略仅在特定单一资产上拟合；
- 同一套因子参数必须同步跑通：
  - **加密市场**: BTC, ETH, SOL, BNB, NEAR（7×24 连续交易、高散户杠杆）；
  - **传统商品与外汇**: XAUUSD, USOUSD, GBPUSD（存在隔夜休市跳空、Tick Volume 计数）。
- 若策略仅在特定山寨币盈利而在外汇/商品全面暴死，提示因子对点差/时段鲁棒性极度敏感。

### 2. 样本内外衰减率审计 (IS/OOS Degradation Ratio)
- 计算样本内到样本外的收益衰减率：
  $$\text{Degradation} = 1 - \frac{\text{Mean OOS Return}}{\text{Mean IS Return}}$$
- **健康区间**: $\text{Degradation} \le 75\%$；
- **过拟合警报**: 若 IS 收益年化超 300% 而 OOS 出现连续负收益，判定为参数过拟合，强制介入波动率 Regime（如 $ATR_{Ratio} \in [0.55, 1.80]$）进行开仓剪枝。

### 3. 蒙特卡洛压力测试规范 (Monte Carlo Bootstrap Suite)
- **输入序列**: 仅使用真实样本外（Out-of-Sample）撮合成交记录（Trades JSON）；
- **重采样配置**: 2,000 条仿真路径，单路径采样 200 笔真实交易；
- **强制合规阈值**:
  - **破产概率 (Probability of Ruin, 跌破 50% 本金)**: 必须严格为 **0.00%**；
  - **95% 置信度最大回撤 (95% MDD)**: 应当 $\le 30.0\%$；
  - **99% 极端尾部回撤 (99% Extreme MDD)**: 应当 $\le 35.0\%$；
  - 输出 95% 在险价值（VaR 95%）与条件在险价值（CVaR 95%）。
