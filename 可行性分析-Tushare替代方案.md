# 可行性分析：用 Tushare 数据接口替代 QMT / xtquant

> 分析对象：`AI量化实盘`（项目代号 deepredict）
> 分析目的：评估将现有 QMT（`xtquant`）数据源切换为 Tushare Pro 的可行性
> 文档性质：**可行性分析，不含任何代码改动**（P.S. 结论已于同日落地实施，见文末「附录」）
> 日期：2026-09-23

---

## 一、结论先行

| 问题 | 结论 |
|---|---|
| 能否用 Tushare 替代 QMT 取数？ | ✅ **可以**，日线级（本项目默认 `1d`）完全可替代，架构上天然适配 |
| 改造代价大吗？ | 🟢 **小**。项目已内建"数据源降级"骨架，只需新增一个 provider 分支，**无需重写主流程** |
| 有硬性障碍吗？ | ⚠️ **有 2 个**：① 分钟线（`1m/5m` 等）Tushare 需**单独开权限**（非积分可解决）；② 期权数据完整性需实测 |
| 积分够用吗？ | ✅ 你说的"积分足够"可覆盖：本项目所需接口集中在 **2000～5000 积分**档 |
| **项目有 UI 页面吗？** | ✅ **有**。`visualize.py` 是 Streamlit 网页面板（热力图 + 黑天鹅 4 图 + 指标卡） |
| **运行结果怎么看？** | ✅ 4 个出口：控制台报告 / CSV+CSV+TXT 落盘 / matplotlib 5 宫格图 / Streamlit 网页 |
| **支持回测吗？** | ✅ **支持**，且是内置的。但属于"训练脚本内嵌的简易回测"，非独立回测框架（详见第五节） |

**一句话总结**：**技术上完全可行，改造量约 1～2 天**；真正的取舍不在"能不能"，而在"你是否需要分钟级与实盘交易通道"。

> ✅ **【最新】本方案已落地实施（方案 A：双数据源并存）。**
> 新增 `data_provider.py` 统一数据层，Tushare 为默认数据源、QMT 保留为可选。详见文末「八、实施结果」。

---

## 二、先回答你的三个问题

### 2.1 这个项目有 UI 页面吗？——有

| UI 形式 | 位置 | 说明 |
|---|---|---|
| **Streamlit 网页面板**（主 UI） | `visualize.py` | 唯一的现代 Web UI。标题「📊 大盘预测热力图」，含：① treemap 热力图（层级=行业→标的，面积=收益强度，颜色=预期收益，绿降红升符合 A 股习惯）；② 三张指标卡（最强收益 / 最大亏损 / 多空比）；③ 明细表格；④ **黑天鹅分析区**（真实vs预测均值、真实vs预测VaR5%、分布漂移vs阈值、仓位曲线，共 4 张 Plotly 交互图） |
| **matplotlib 静态图**（隐式 UI） | `main.py` L633–722 | 训练结束后弹出 5 宫格图（`plt.show()`）。不是网页，是本地窗口 |
| 命令行报告（文本 UI） | `predict.py` | 8 段格式化终端报告，带 ANSI 颜色（风险等级 红/黄/绿） |

> ⚠️ 注意：`visualize.py` 里 `main()` 被定义了**两次**（L88 与 L180），Python 后定义覆盖前者，实际生效的是 L180 版本（含黑天鹅区）。L88 那个是死代码。

### 2.2 运行结果如何查看？——4 个出口

**出口 A：控制台输出**
- `predict.py` → 8 段报告：全市场统计 / 风险指标 VaR / 市场情绪 / 风险评分 / Top-N / Bottom-N / 交易信号 / 保存提示
- `main.py` → `=== 策略绩效统计 ===`（初始资金、最终资金、总收益率、夏普、最大回撤、交易成本、交易次数）

**出口 B：结构化文件落盘**

| 文件 | 内容 |
|---|---|
| `predict_output/prediction_YYYY-MM-DD.csv` | 逐标的：代码 / 名称 / 预期收益 |
| `predict_output/summary_YYYY-MM-DD.txt` | 单日摘要：均值、标准差、VaR5%/1%、上涨比例、交易信号 |
| `predict_output/black_swan_analysis.csv` | 逐日全指标：真实/预测均值、VaR5%、分布漂移、漂移阈值、仓位、组合价值 |
| `predict_output/portfolio_stats.txt` | 回测绩效汇总 |
| `hedge_data/hedge_results.csv` 等 4 个 CSV | 对冲分析结果、相关系数矩阵、期权信息、收益率 |
| `models/distribution_predictor.pth` + `config.npy` | 训练好的模型权重 + 配置（含标准化参数） |

**出口 C：图表**
- `main.py` 结束时的 5 宫格图（直接弹窗）

**出口 D：Streamlit 网页**
```bash
uv run streamlit run visualize.py     # 浏览器打开 http://localhost:8501
```
> 前提：先跑过 `predict.py` 生成 CSV，否则页面报「输出目录不存在」。

### 2.3 支持回测吗？——支持，但是"内嵌式简易回测"

**支持，位置在 `main.py`，不是独立脚本。**

- **回测引擎**：`PositionController` 类（`main.py` L102–248）
- **回测区间**：仅测试集（数据按 7:1.5:1.5 划分，测试集占最后 15%）
- **已实现**：仓位档位（0/0.5/1.0）、止损 -5%、止盈 +10%、单日调仓上限 30%、交易成本万 1、最小持仓 5 天、夏普比率、最大回撤、交易次数
- **输出的绩效**：总收益率 / 夏普 / 最大回撤 / 总交易成本 / 交易次数

**⚠️ 但有 3 个重要限制，务必知情：**

1. **不是独立回测框架**：回测逻辑与训练逻辑写在同一个文件、同一次运行里，**无法只回测不训练**（每次都要重训模型）。
2. **`config.py` 里的 `BacktestConfig` 是"死配置"**：我核对了全部 6 个脚本的 import —— `BacktestConfig`、`ScorecardConfig`、`FactorConfig`、`QMTTraderConfig` 以及 `load_symbols_from_csv()` **全部未被任何脚本引用**。只有 `DataConfig` 和 `QMTConfig` 真正生效。回测参数实际硬编码在 `main.py` L587–596，与 `config.py` 里的 `BacktestConfig` 同名参数**不联动**（改了 config 不生效）。
3. **无基准对比**：`BacktestConfig.compare_indices = ['510300.SH','510500.SH','510050.SH']` 定义了但没用，回测结果**没有与沪深300等指数做超额对比**。
4. **无滑点/无涨跌停/无停牌处理**：只有固定万 1 成本，实盘会有偏差。

> 📌 结论：**自用研究够用，若要对标正式策略评估，建议补齐基准对比与独立回测脚本。**

---

## 三、Tushare 替代方案：逐调用映射

项目对 QMT 的依赖只有 **6 类调用**，全部有 Tushare 对应接口。

| # | 现有 QMT 调用（位置） | 用途 | Tushare 替代接口 | 积分门槛 |
|---|---|---|---|---|
| 1 | `xtdata.download_history_data` + `get_market_data_ex(field_list=["time","open","high","low","close","volume"])`<br>`train_data_generator.py` L137–155 | ETF 日线 OHLCV | `pro.fund_daily(ts_code, start_date, end_date)` | **2000～5000** |
| 2 | 同上但 `dividend_type='front'`（前复权）<br>`etf_hedge_analysis.py` L147–156 | ETF 复权行情 | `pro.fund_adj()` 取复权因子 + 自行换算；或 `ts.pro_bar(asset='FD', adj='qfq')` | **600**（fund_adj）～5000 |
| 3 | `xtdata.get_stock_list_in_sector('沪深A股')`<br>`predict.py` L543、`train_data_generator.py` L86 | 板块成分股 | `pro.stock_basic()` / `pro.index_member()` | **120** / 2000 |
| 4 | `xtdata.get_instrument_detail(symbol)`<br>`predict.py` L147 | 标的中文名 | `pro.fund_basic()`（含 ETF 名称）→ 可据此生成 `ETF.json` | **2000** |
| 5 | `xtdata.get_option_list(undl_code, dedate, opttype, isavailavle)`<br>`etf_hedge_analysis.py` L200–205 | 期权合约列表 | `pro.opt_basic(exchange, call_put=...)` 按标的 + 到期日过滤 | **2000** |
| 6 | `get_market_data_ex`（对期权合约）<br>`etf_hedge_analysis.py` L214–223 | 期权日线行情 | `pro.opt_daily(ts_code / trade_date / exchange)` | **2000～5000** |
| 7 | `xtdc.set_token / set_quote_time_mode_v2 / set_data_home_dir / init(False) / listen(port=58615)`<br>4 个脚本的头部 | QMT 数据中心常驻连接 | **不需要**。Tushare 是 HTTP REST，无连接态，反而更简单 | — |

### 3.1 一个重要的积分口径提醒

Tushare 官方**「数据接口列表」页与各接口文档页给出的门槛存在不一致**，我两处都核对了：

| 接口 | 官方接口列表页（doc_id=108） | 接口详情页 |
|---|---|---|
| `fund_daily` 场内基金日线 | 2000 | 「至少 5000」 |
| `fund_adj` 基金复权因子 | 5000 起 | 「600 可调取」 |
| `opt_basic` 期权合约列表 | 2000 起 | 「至少 5000」 |
| `opt_daily` 期权日线 | 5000 起 | 「至少 2000」 |

> **建议**：以**实测调用**为准。你积分足够，这大概率不构成障碍；但首次接入时请逐个接口试调，不要按文档门槛做承诺式规划。
> 另：**积分本身受"每分钟频次 + 每天总量"双重限制**，批量拉 63 只 ETF × 十年数据需要做**分片循环 + 本地缓存**（项目已有 pickle 缓存机制，可直接复用）。

### 3.2 不可替代 / 需注意的能力缺口

| 缺口 | 严重度 | 说明与应对 |
|---|---|---|
| **分钟线（1m/5m/15m/30m/1h）** | 🔴 高 | Tushare 的分钟数据（`stk_mins`/`opt_mins`）属于**"需单独开权限"**类别，**与积分无关**，需另行申请。而项目 `config.py` 的 `frequency` 字段支持分钟级 —— **若你要用分钟级预测，Tushare 是硬门槛**；若只用默认日线（`1d`），无影响 |
| **深交所期权（159915 创业板ETF期权）覆盖度** | 🟡 中 | `opt_basic` / `opt_daily` 的 `exchange` 参数支持 `SZSE`，理论覆盖；但需**实测确认合约列表是否完整**再投产 |
| **Tick / 实时行情** | 🟡 中 | 项目本身不使用 Tick（回测用日线），影响有限。但**实时盘中预测**（`predict.py --time`）依赖的时点数据，Tushare 日线接口给不了，需另接实时源 |
| **实盘下单通道** | 🟡 中 | `QMTTraderConfig` 定义了但**代码中从未使用**（无下单逻辑）。所以"替代 QMT 会损失下单能力"——**实际上现在也没有**。项目是纯研究/回测定位 |
| **数据接口稳定性** | 🟢 低 | QMT 是本机常驻数据中心，Tushare 走公网 HTTP，需处理限流重试（加 `time.sleep` + 指数退避即可） |

---

## 四、改造工作量评估

### 4.1 为什么说改造很小——架构已就绪

**关键发现**：项目作者已经内建了**"数据源降级"骨架**，这是最大的利好。

- `train_data_generator.py` L115–117：`if not QMT_AVAILABLE: return self._load_local_data()`
- `etf_hedge_analysis.py` L123–125：`if not QMT_AVAILABLE: return self._load_local_etf_data(...)`
- 4 个脚本头部都有 `try: from xtquant import ... except ImportError: QMT_AVAILABLE = False`

也就是说：**当 QMT 不可用时，代码已有明确的降级分支**。接入 Tushare 的最优做法不是"替换 QMT 代码"，而是**在降级分支旁边新增一个 `TushareProvider`**，把"取 OHLCV"抽象成一个统一函数：

```python
# 伪代码：改造后的取数入口
def get_ohlcv(symbol, start, end, freq='1d'):
    if provider == 'qmt':      return _from_qmt(...)        # 现有逻辑
    elif provider == 'tushare': return _from_tushare(...)   # 新增
    elif provider == 'csv':     return _from_local_csv(...) # 现有逻辑
```

### 4.2 具体改造清单

| 文件 | 改造点 | 行号定位 | 工作量 |
|---|---|---|---|
| `config.py` | 新增 `TushareConfig`（token 走环境变量 `TUSHARE_TOKEN`）；`QMTConfig` 保留 | L145–156 | 🟢 小 |
| `train_data_generator.py` | ① QMT 初始化块（L19–47）增加 Tushare 分支；② `_download_data`（L113–196）内 `xtdata.download_history_data` + `get_market_data_ex`（L137–155）替换为 `pro.fund_daily`；③ 补 `fund_adj` 复权换算 | L19–47、L113–196 | 🟡 中 |
| `predict.py` | ① 初始化块（L22–48）；② `get_instrument_detail`（L147）→ `fund_basic`；③ `get_stock_list_in_sector`（L543）→ `stock_basic` | L22–48、L147、L543 | 🟢 小 |
| `etf_hedge_analysis.py` | ① 初始化块（L21–49）；② ETF 取数（L144–156）；③ **期权链路改造**（L191–249：`get_option_list` → `opt_basic`，期权行情 → `opt_daily`） | L21–49、L144–156、L191–249 | 🔴 **最大工作量** |
| `main.py` / `visualize.py` / `pyproject.toml` | **无需改动**（不直接接触数据源）；仅 `pyproject.toml` 增加 `tushare` 依赖 | — | 🟢 极小 |

**总体评估**

| 阶段 | 内容 | 复杂度 |
|---|---|---|
| P0 最小可用 | ETF 日线 OHLCV 打通（训练+预测全链路） | 🟢 简单，约半天 |
| P1 完整性 | 复权因子、标的名称、`ETF.json` 自动生成 | 🟢 简单，约半天 |
| P2 难点 | 期权链路（opt_basic + opt_daily 重写） | 🟡 中等，约半天～1 天 |
| P3 健壮性 | 限流退避、分片循环、缓存复用、对账验证 | 🟡 中等 |

> **结论：核心链路（训练+预测+回测+可视化）约 1 天可打通；含期权对冲的全功能约 2 天。**
> 若**放弃期权对冲模块**（`etf_hedge_analysis.py` 是独立功能，不影响主策略），改造量骤降到 **半天**。

---

## 五、推荐落地方案

### 方案 A（推荐 ⭐）：双数据源并存，Tushare 优先 + QMT 可选

- 保留 QMT 代码路径不动，新增 Tushare provider，通过 `config` 开关切换。
- **理由**：① 改造风险最低（不动现有已验证逻辑）；② 你积分充足，Tushare 无需本机常驻数据中心，跨机器/云端部署更容易；③ 未来若要接实盘，QMT 通道仍在。
- **代价**：代码里多一层抽象。

### 方案 B：Tushare 全量替换，删除 QMT 依赖

- **理由**：代码最干净，`pyproject.toml` 可移除 `xtquant`，**开源到 GitHub 时零私有依赖**（对个人项目展示更友好）。
- **代价**：损失 QMT 数据源；`pyproject.toml` 的 `xtquant>=250516.1.1` 依赖本来就是 Linux/Mac 装不上的（需 QMT 环境），移除后**跨平台性大幅提升**。

### 方案 C：不改造，仅补一个 Tushare 备用取数脚本

- 写一个 `fetch_tushare_to_csv.py`，先把 Tushare 数据落成项目已支持的 `data/etf_hist_data.csv` 格式，让现有 `_load_local_data()`（降级分支）直接吃。
- **理由**：**零侵入**，一行现有代码都不用改，最快见效。
- **代价**：无法用分钟级；期权模块走不通。

> **给你的建议**：若目标是"**上传 GitHub 展示 + 个人研究**"，选 **方案 B**（去掉 xtquant 私有依赖，项目立刻变成任何人 clone 就能跑的开源项目）；若还要兼顾实盘，选 **方案 A**。若只想**先验证效果**，用 **方案 C** 试水半天。

---

## 六、附：QMT 依赖完整清单（供改造核对）

| 文件 | 行号 | 调用 |
|---|---|---|
| `train_data_generator.py` | L19–47 | `from xtquant import xtdata, xtdatacenter`；`set_token` / `set_quote_time_mode_v2` / `set_data_home_dir` / `set_index_mirror_enabled` / `set_future_realtime_mode` / `init(False)` / `listen(58615)` |
| | L86 | `xtdata.get_stock_list_in_sector(sector)` |
| | L137–142 | `xtdata.download_history_data(...)` |
| | L148–155 | `xtdata.get_market_data_ex(field_list=["time","open","high","low","close","volume"], ...)` |
| `predict.py` | L22–48 | 同上初始化块 |
| | L147 | `xtdata.get_instrument_detail(symbol)` |
| | L543 | `xtdata.get_stock_list_in_sector('沪深A股')` |
| `etf_hedge_analysis.py` | L21–49 | 同上初始化块 |
| | L144 | `xtdata.download_history_data(code, period)` |
| | L147–156 | `get_market_data_ex(..., dividend_type='front', fill_data=True)` |
| | L200–205 | `xtdata.get_option_list(undl_code, dedate, opttype, isavailavle=True)` |
| | L211、L214–223 | 期权 `download_history_data` + `get_market_data_ex` |

**共 3 个文件、约 8 处调用点需要改造。**

---

## 七、风险提示（改造时务必注意）

1. **复权口径必须一致**：QMT 用 `dividend_type='front'`（前复权），Tushare 需显式用 `fund_adj` 换算。**若口径不一致，特征分布会整体偏移，模型直接失效**。建议改造后用一小段重叠区间做对账（同期同标的收益率差应 < 1e-6）。
2. **`data_mean`/`data_std` 会变**：换数据源后标准化参数必须**重新生成**（`data_mean.npy` / `data_std.npy`），否则模型输入尺度错位。**换源后必须重训模型**。
3. **`ETF.json` 可自动生成**：改完后顺手用 `fund_basic` 生成 `ETF.json`（当前缺失），代码里 `load_instrument_names()` 会自动读取。
4. **不要提交 token**：Tushare token 写入环境变量，`.gitignore` 排除 `.env`。
5. **积分频次**：63 只 ETF × 10 年 ≈ 15 万条日线，受单次 5000 行限制，**必须按代码循环 + 本地缓存**（项目已有 `cache/` 机制，复用即可）。

---

## 八、实施结果（方案 A 已落地）

> 本节记录 2026-09-23 的实际实施与验证结果，供后续维护参考。

### 8.1 新增与改动文件

| 文件 | 类型 | 说明 |
|---|---|---|
| `data_provider.py` | 🆕 新增 | **统一数据层**，含 `TushareProvider` / `QMTProvider` / `CSVProvider` + `get_provider()` 工厂 |
| `config.py` | ✏️ 改动 | 新增 `TushareConfig`（token / sleep / adjust）与 `DataSourceConfig`（provider 开关） |
| `train_data_generator.py` | ✏️ 改动 | 移除 QMT 硬依赖，取数改走 `provider.get_ohlcv()`；板块成分股改走 `provider.get_stock_list_in_sector()` |
| `predict.py` | ✏️ 改动 | 标的中文名改走 `provider.get_instrument_names()`；股票池回落改走 provider |
| `etf_hedge_analysis.py` | ✏️ 改动 | `_download_etf_data` 与 `_download_option_data` 改走 provider |
| `pyproject.toml` | ✏️ 改动 | `tushare` 移入主依赖；`xtquant` 移入 `[project.optional-dependencies].qmt` |
| `config.example.py` | ✏️ 改动 | 同步新增 `DataSourceConfig` / `TushareConfig` 模板 |
| `ETF.json` | 🆕 生成 | 63 只 ETF 中文名映射（原项目缺失） |
| `ETF清单.md` | 🆕 新增 | 63 只 ETF 全名单 + 行业分类 |

**核心设计**：取数逻辑收敛为统一接口，**换数据源 = 改 `config.py` 一行**，训练/回测/预测主流程代码完全不变。

### 8.2 实测验证结果（全部通过 ✅）

| 验证项 | 结果 |
|---|---|
| Tushare token 连通性 | ✅ `trade_cal` 正常 |
| 63 只 ETF 名称匹配 | ✅ **63 / 63 = 100%** |
| `fund_daily`（ETF 日线） | ✅ 沪市 + 深市均正常 |
| `fund_adj`（复权因子） | ✅ 正常 |
| `pro_bar(asset='FD', adj='qfq')` | ✅ 直出前复权，**无需手工换算** |
| `fund_basic(market='E')` | ✅ 返回 2955 只基金 |
| `stock_basic`（A股列表） | ✅ 返回 5568 只 |
| `opt_basic`（期权合约） | ✅ SSE 12000+ 条 / SZSE 8000+ 条 |
| `opt_daily`（期权日线） | ✅ 按到期月过滤后正常取数 |
| 宽基期权覆盖 | ✅ 510300 / 510050 / 510500 / 588000 / **159915（深市创业板）** |
| **数据生成** | ✅ `training_data.npy` 形状 **(2541, 1260)**，区间 2016-01-05 ~ 2026-06-24 |
| **训练 + 回测** | ✅ GPU (RTX 3070 Laptop 8.59GB) 正常，产出模型 + 黑天鹅 CSV + 绩效 |
| **单日预测** | ✅ 风险评分 / Top-N / Bottom-N / 交易信号 / 中文名全部正常 |
| **期权对冲分析** | ✅ 63 行业 ETF × 5 宽基 ETF，相关矩阵 + Beta + ATM 期权定位全部产出 |

### 8.3 重要技术发现（对原方案的修正）

1. **`pro_bar(asset='FD', adj='qfq')` 可直接出前复权行情** —— 原方案预判需要 `fund_daily` + `fund_adj` 手工换算，实测发现 Tushare 原生支持，**大幅简化了实现**（手工换算逻辑保留为兜底分支）。
2. **期权标的映射有官方字段** —— `opt_basic.opt_code` 形如 `OP510300.SH`，即 `'OP' + 标的代码`，**不依赖名称模糊匹配**，比原方案预判更可靠。
3. **深市期权确实覆盖** —— 原方案标注为「需实测确认」，实测 SZSE 覆盖 `159901 / 159915 / 159919 / 159922`，**159915 创业板ETF期权可用**。
4. **积分门槛实测无忧** —— 你账号的权限足以调用全部所需接口，**文档中的门槛差异未构成障碍**。
5. **分钟级仍然是唯一硬约束** —— Tushare 分钟数据需单独开权限，`TushareProvider` 已做显式提示并安全跳过；需要分钟级时切回 `provider='qmt'`。

### 8.4 使用方式

```bash
# 默认走 Tushare
export TUSHARE_TOKEN=your_token     # 或在 config.py 的 TushareConfig.token 填写

# 切回 QMT
# config.py -> DataSourceConfig.provider = 'qmt'

# 自动探测（tushare -> qmt -> csv）
# config.py -> DataSourceConfig.provider = 'auto'
```

---

*本文档的可行性论证与实施结果均为 2026-09-23 记录。*
