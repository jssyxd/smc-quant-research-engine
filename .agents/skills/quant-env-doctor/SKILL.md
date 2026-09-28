# Skill: quant-env-doctor

## 用途与定位
量化环境自检、缺失资源智能感知、数据源定向拉取与依赖自愈工具。在开启任何回测或因子研究任务前强制运行。

## 探查清单与标准自愈路由

### 1. 硬件与系统内存探查
- 执行内存总量与可用内存检查：`psutil.virtual_memory().available`。
- 若系统可用内存 $< 4.0\text{ GB}$，自动设置数据分块保护参数：`chunk_size = 50000` 并强制每轮回测调用 `gc.collect()`。

### 2. 回测数据自动发现与拉取路由
- **目标数据库**: `https://github.com/JasonleeQAQ/multi-asset-ohlcv`
- **拉取策略**:
  1. 检查本地 `data/raw_extracted/{SYMBOL}/` 是否存在指定时间周期的 Parquet 文件；
  2. 若缺失，从 GitHub Releases `1.0.0` 定向下载对应压缩包：
     `https://github.com/JasonleeQAQ/multi-asset-ohlcv/releases/download/1.0.0/{SYMBOL}USDTraw_data.zip`
     （传统资产: `xauusdraw_data.zip`, `usousdraw_data.zip`, `gbpusdraw_data.zip`）；
  3. **严格流式内存解压**: 仅提取指定周期文件（如 `*15m*`, `*1h*`），严禁全量释放 1m 级大文件以防爆内存。

### 3. 回测框架与算法工具路由
- **NautilusTrader**: `https://github.com/nautechsystems/nautilus_trader`
  - 检查命令: `python3 -c "import nautilus_trader"`
  - 适用场景: 纳秒级高精度订单簿事件驱动、真实挂单队列模拟。
- **QuantCell**: `https://github.com/pengwow/QuantCell`
  - 检查命令: `python3 -c "import axon_bridge"` 或底层 `axon_quant`。
  - 适用场景: 高性能资金曲线与多品种横截面调度。
- **微软 Qlib 因子库**: `https://github.com/microsoft/qlib`
  - 适用场景: 高维 Alpha 因子库构建、多因子 Confluence 特征工程。
- **微软 RD-Agent 自动化研发框架**: `https://www.microsoft.com/en-us/research/articles/rd-agent-quant/`
  - 适用场景: 自动化构建因子假设、生成验证代码、闭环评估反馈。
