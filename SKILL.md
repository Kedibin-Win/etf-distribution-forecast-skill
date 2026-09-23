---
name: etf-distribution-forecast
description: ETF 全市场收益分布预测与状态择时技能。用 Transformer 预测 A 股 63 只主流 ETF 的次日横截面收益分布，输出风险指标（VaR/偏度/峰度）、0-100 风险评分、交易信号、三档仓位建议，并以 JS 散度漂移作为黑天鹅闸门。当用户需要「市场择时」「仓位管理」「系统性风险预警」「ETF 轮动」或「用分布而非点位做风控」时使用。
version: 1.0.0
author: Kedibin-Win
license: MIT
agent_created: true
tags:
  - quant
  - etf
  - transformer
  - market-timing
  - risk-management
  - backtest
---

# 🌊 ETF 分布预测与状态择时（etf-distribution-forecast）

> **核心命题**：不预测"哪只 ETF 会涨"，而预测"明天整个市场的收益分布长什么样"。
> 分布可做统计推断（VaR / 偏度 / 峰度），单点不能 —— 这是"度量风险"与"猜涨跌"的分野。

---

## 一、能力边界（先看清楚）

### ✅ 能做什么

| 能力 | 对应脚本 | 产出 |
|---|---|---|
| 生成 63 只 ETF × 20 维特征面板 | `train_data_generator.py` | `data/*.npy` |
| 训练 Transformer 分布预测模型 | `main.py` | `models/distribution_predictor.pth` |
| 测试集回测 + 绩效统计 | `main.py` | 收益率 / 夏普 / 最大回撤 |
| 黑天鹅漂移诊断图 | `main.py` | `predict_output/black_swan_analysis.csv` + 5 宫格图 |
| 指定日期/时点的市场预测 | `predict.py` | 风险评分 + 交易信号 + Top/Bottom N |
| Web 可视化面板 | `visualize.py` | Streamlit 热力图 + 黑天鹅四联图 |
| 行业 ETF × 宽基 ETF 期权对冲比例 | `etf_hedge_analysis.py` | `hedge_data/*.csv` |

### ❌ 不能做什么（硬边界，不要越界承诺）

| 不可做 | 原因 |
|---|---|
| **实盘下单** | 项目**未实现任何交易通道**（`QMTTraderConfig` 定义了但从未被 import） |
| **单标的择时** | 模型输出的是横截面分布，不是单只标的的时序预测 |
| **保证收益 / 投顾建议** | 回测为历史模拟，不代表未来；本技能不构成投资建议 |
| **分钟级预测（默认不可用）** | `config.py` 支持分钟频率，但数据源需相应权限（QMT 需环境、Tushare 需单独开权限） |
| **免重训的快速回测** | 回测内嵌在训练脚本里，改参数需重训模型 |

### ⚠️ 必须向用户声明的三件事

1. **换数据源必须重训模型** —— `data_mean` / `data_std` 会变，标准化尺度错位会直接导致模型失效。
2. **复权口径必须一致**（统一前复权 `qfq`），否则特征整体偏移。
3. **回测有已知缺陷**：无基准对比、无滑点、无涨跌停/停牌处理，样本外仅最后 15%。

---

## 二、七步工作流

```
0 · 选数据源          config.DataSourceConfig   → tushare / qmt / csv / auto
1 · 生成特征面板      train_data_generator.py   → data/*.npy
2 · 训练分布模型      main.py                   → models/*.pth
3 · 回测 + 出绩效     main.py（内嵌）           → portfolio_stats.txt
4 · 黑天鹅诊断        main.py（内嵌）           → black_swan_analysis.csv
5 · 单日预测          predict.py --date         → prediction_*.csv / summary_*.txt
6 · 可视化            visualize.py              → Web UI
7 · （可选）对冲分析  etf_hedge_analysis.py     → hedge_data/*.csv
```

### 命令参考

```bash
# 0. 环境 + 数据源
uv sync
export TUSHARE_TOKEN=your_token      # 或用 QMT：改 config.provider='qmt'

# 1. 生成训练数据
uv run python train_data_generator.py [--force-refresh] [--window-size N]

# 2. 训练 + 回测（--amp 省显存）
uv run python main.py [--epochs 1000] [--batch-size 128] [--lr 1e-5] [--amp] [--no-early-stop]

# 3. 单日预测
uv run python predict.py --date YYYY-MM-DD [--time HH:MM:SS] [--top-n 10] [--bottom-n 10]

# 4. 可视化
uv run streamlit run visualize.py

# 5. 期权对冲分析（可选）
uv run python etf_hedge_analysis.py [--start-date 2025-01-01] [--end-date YYYY-MM-DD]
```

> **依赖顺序严格**：`config` → `train_data_generator` → `main` → `predict` → `visualize`。
> `etf_hedge_analysis` 独立，仅依赖 `ETF.json`（可选）与数据源。

---

## 三、核心口径契约（改代码前必读）

### 3.1 特征契约（20 维，顺序不可变）

`train_data_generator.py` 与 `main.py` / `predict.py` **各自硬编码了一份 `FEATURES` 列表**，必须三处完全一致：

```
['open_return', 'high_return', 'low_return', 'close_return',
 'volume_change', 'range_ratio', 'body_ratio', 'upper_shadow', 'lower_shadow',
 'ma5', 'ma10', 'ma20', 'ma60', 'std5', 'std20', 'rsi', 'macd', 'macd_signal',
 'correlation', 'market_rank']
```

**张量形状约束**：`数据列数 = ETF 数量 × 20`。ETF 数量从 `config.py` 的 `etf_list` 推断（当前 63 只 → 1260 维）。
**改 ETF 清单后必须重新生成数据 + 重训模型**，否则 `predict.py` 会报「配置股票池数量与训练数据数量不一致」。

### 3.2 模型契约

```
输入:  (batch, 60, 1260)       # 窗口 60 × (63 ETF × 20 特征)
输出:  (batch, 1260)           # 次日全市场状态
```
- Encoder：3 层，`d_model=256`，`nhead=8`，`dim_feedforward=1024`，`GELU`，`pre-norm`
- 训练：AdamW(`lr=1e-5`, `wd=1e-2`) + CosineAnnealing(200 步 warmup)，梯度裁剪 0.3，早停 patience=200
- 划分：`7 : 1.5 : 1.5`（train / val / test）

### 3.3 风控契约（真实参数在 `main.py`，不在 `config.py`）

| 参数 | 值 |
|---|---|
| 仓位档位 | `0 / 0.5 / 1.0` |
| 确认天数 | 2 日 |
| 满仓阈值 / 半仓阈值 | 0.0005 / 0.0001 |
| 止损 / 止盈 | -5% / +10% |
| 单日调仓上限 | 30% |
| 交易成本 | 0.0001 |

> ⚠️ **`config.py` 的 `BacktestConfig` 是死配置** —— 6 个脚本均未 import。改它不生效。同样未生效的还有 `FactorConfig` / `ScorecardConfig` / `QMTTraderConfig` / `load_symbols_from_csv()`。

### 3.4 风险评分契约（0-100）

| 维度 | 触发 | 分 |
|---|---|---|
| 收益均值 | `< -0.5%` / `< 0` | 30 / 10 |
| 波动率 | `> 2%` / `> 1.5%` | 30 / 15 |
| 尾部 | `VaR5 < -5%` / `< -3%` | 30 / 15 |
| 广度 | `上涨比例 < 30%` / `< 45%` | 20 / 10 |
| 峰度 | `> 3` | 15 |
| 偏度 | `< -0.5` | 10 |

等级：`≥70 HIGH` / `40-70 MEDIUM` / `<40 LOW`
信号：`RISK_OFF` / `STRONG_BUY` / `CAUTIOUS_BUY` / `WEAK_BUY` / `HOLD` / `SELL`

---

## 四、数据源（方案 A：双数据源并存）

取数逻辑统一收敛在 `data_provider.py`。**切换数据源只改 `config.py` 一行**，主流程代码零改动。

```python
# config.py
class DataSourceConfig:
    provider = 'tushare'   # 'tushare' | 'qmt' | 'csv' | 'auto'
```

| 数据源 | 类 | 分钟级 | 期权 | 说明 |
|---|---|---|---|---|
| **Tushare** ⭐ | `TushareProvider` | ❌ | ✅ | **默认**。跨平台，`pro_bar(asset='FD', adj='qfq')` 直出前复权 |
| **QMT / xtquant** | `QMTProvider` | ✅ | ✅ | 可选。需本机 QMT 环境，保留分钟级与实时能力 |
| **本地 CSV** | `CSVProvider` | ❌ | ❌ | 兜底。读 `data/etf_hist_data.csv` |
| `auto` | — | — | — | 依次尝试 `tushare → qmt → csv` |

**统一接口**（三源返回结构完全一致）：
```python
get_ohlcv(symbols, start, end, frequency='1d', adjust='qfq')  -> {code: DataFrame[date,open,high,low,close,volume]}
get_instrument_names(symbols)                                  -> {code: 中文名}
get_option_contracts(undl_code, dedate, opttype)               -> [code, ...]
get_option_ohlcv(codes, start, end)                            -> {code: DataFrame}
get_stock_list_in_sector(sector)                               -> [code, ...]
```

**QMT → Tushare 接口映射**

| 用途 | Tushare 接口 |
|---|---|
| ETF 日线（前复权） | `ts.pro_bar(asset='FD', adj='qfq')` |
| ETF 日线（不复权） | `pro.fund_daily` |
| 复权因子（兜底） | `pro.fund_adj` |
| 标的中文名 | `pro.fund_basic(market='E')` |
| A 股列表 | `pro.stock_basic` |
| 期权合约列表 | `pro.opt_basic`（`opt_code` = `OP<标的代码>`） |
| 期权日线 | `pro.opt_daily` |

**本地 CSV 期望列**：`symbol, timestamp, open, high, low, close, volume`

**已实测验证**：63/63 ETF 名称匹配；`fund_daily` / `fund_adj` / `fund_basic` / `opt_basic` / `opt_daily` 权限全部正常；宽基期权覆盖 510300 / 510050 / 510500 / 588000 / **159915（深市）**；四个脚本全部端到端跑通。

> ⚠️ **分钟级限制**：Tushare 分钟数据需**单独开权限**（与积分无关）。需分钟级时把 `provider` 切回 `'qmt'`。
>
> 📄 完整论证见 [`可行性分析-Tushare替代方案.md`](./可行性分析-Tushare替代方案.md)。
> 📋 63 只 ETF 全名单见 [`ETF清单.md`](./ETF清单.md)。

---

## 五、输出契约

| 产物 | 路径 | 用途 |
|---|---|---|
| 特征面板 | `data/training_data.npy` + `_dates` + `data_mean/std.npy` | 训练输入 |
| 模型 | `models/distribution_predictor.pth` + `config.npy` | 推理 |
| 预测明细 | `predict_output/prediction_YYYY-MM-DD.csv` | 逐标的预期收益 |
| 预测摘要 | `predict_output/summary_YYYY-MM-DD.txt` | 单日风险摘要 |
| 黑天鹅面板 | `predict_output/black_swan_analysis.csv` | 逐日全指标 |
| 绩效汇总 | `predict_output/portfolio_stats.txt` | 回测结果 |
| 对冲结果 | `hedge_data/*.csv` | 相关性 / 期权 / 收益率 |

---

## 六、护栏与禁忌

| 护栏 | 说明 |
|---|---|
| 🔒 **不承诺收益** | 任何输出必须附带"回测不代表未来"的声明 |
| 🔒 **不声称可实盘** | 项目无下单通道，切勿暗示可自动交易 |
| 🔒 **换源必重训** | 换数据源后必须重新生成 `.npy` 与重训 `.pth`，不得复用旧模型 |
| 🔒 **复权口径统一** | 全程前复权，混用会静默污染特征 |
| 🔒 **不提交凭证** | `config.py` / `.env` 不入库，只提交 `config.example.py` |
| 🔒 **诚实标注缺陷** | 回测无基准/无滑点/无停牌处理，不得声称"已充分验证" |
| 🔒 **预设参数不擅自调** | 阈值是调优结果，改动前先说明影响面 |

---

## 七、常见故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `QMT模块导入失败` | 未装 `xtquant` | 正常，自动降级本地 CSV |
| `训练数据不存在: data/training_data.npy` | 未生成特征 | 跑 `train_data_generator.py` |
| `模型文件不存在: models/*.pth` | 未训练 | 跑 `main.py` |
| `配置股票池数量(N)与训练数据数量(M)不一致` | `config.py` ETF 清单变了 | 重新生成数据 + 重训 |
| 预测日期不在训练数据中 | 超出数据范围 | 用数据区间内日期，或用最后一天的下一天做次日推演 |
| Streamlit 报「输出目录不存在」 | 未生成预测 | 先跑 `predict.py` |
| 热力图无黑天鹅图 | 未训练 | 先跑 `main.py` |
| CPU 训练极慢 | 无 CUDA 版 PyTorch | 确认安装 CUDA 版；8GB 显存建议 `--amp` |

---

## 八、参考

- [`README.md`](./README.md) —— 项目展示与完整用法
- [`使用说明.md`](./使用说明.md) —— 逐文件功能详解
- [`可行性分析-Tushare替代方案.md`](./可行性分析-Tushare替代方案.md) —— 数据源替换论证

---

*MIT License · Copyright (c) 2026 Kedibin-Win*
