#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF策略回测项目配置文件 —— 模板示例
=====================================
使用方式：
    cp config.example.py config.py
然后按注释填写你自己的凭证。config.py 已被 .gitignore 忽略，不会入库。

⚠️ 安全提示：请勿把包含真实 token / 账户 ID 的 config.py 提交到公开仓库。
"""

import os
import pandas as pd


# =========================================================
# 数据源凭证（优先从环境变量读取，避免硬编码）
# =========================================================
#
# QMT 数据中心 token：
#   PowerShell:  $env:QMT_TOKEN = "your_token"
#   Bash:        export QMT_TOKEN=your_token
#
# Tushare token（若改用 Tushare 数据源）：
#   export TUSHARE_TOKEN=your_token
#
QMT_TOKEN = os.environ.get('QMT_TOKEN', '')          # ← 留空则无法连接 QMT，会降级读本地 CSV
TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '')  # ← 留空则不可用 Tushare


def load_symbols_from_csv(csv_path):
    """从CSV文件加载股票代码"""
    if not os.path.exists(csv_path):
        print(f"警告: CSV文件不存在: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    if 'symbol' in df.columns:
        return df['symbol'].tolist()
    elif 'code' in df.columns:
        return df['code'].tolist()
    else:
        print(f"警告: CSV文件中找不到 'symbol' 或 'code' 列")
        return []


# 数据获取配置
class DataConfig:
    start_date = '2016-01-01'  # 开始日期
    end_date = '2026-06-24'    # 结束日期
    frequency = '1d'           # 数据频率

    input_frequency = '1d'    # 输入数据频率
    input_window = 60         # 输入窗口大小
    target_frequency = '1d'   # 目标数据频率

    # 股票池配置
    stock_pool = {
        'type': 'etf',  # 'sector' 板块成分股, 'custom' 自定义股票池, 'etf' ETF列表, 'csv' 从CSV文件读取
        'sectors': ['沪深A股'],  # 板块列表
        'custom_symbols': [],  # 自定义股票池
        'csv_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'custom_stocks.csv'),
        'etf_list': [
            "588200.SH", "512170.SH", "159995.SZ", "512000.SH", "512480.SH",
            "512010.SH", "159819.SZ", "512660.SH", "562500.SH", "512690.SH",
            "159928.SZ", "512760.SH", "512710.SH", "512800.SH", "515790.SH",
            "159869.SZ", "512070.SH", "515050.SH", "512200.SH", "515070.SH",
            "512670.SH", "510230.SH", "159851.SZ", "512400.SH", "516160.SH",
            "515030.SH", "515220.SH", "159865.SZ", "159611.SZ", "516510.SH",
            "515170.SH", "515120.SH", "159859.SZ", "159766.SZ", "159998.SZ",
            "159852.SZ", "159755.SZ", "512980.SH", "515400.SH", "159516.SZ",
            "515210.SH", "159825.SZ", "159732.SZ", "159939.SZ", "562570.SH",
            "159997.SZ", "159996.SZ", "159638.SZ", "159837.SZ", "516780.SH",
            "159206.SZ", "159840.SZ", "516300.SH", "510500.SH", "510050.SH",
            "159745.SZ", "159698.SZ", "516020.SH", "516570.SH", "159326.SZ",
            "561380.SH", "562550.SH", "159647.SZ"
        ]  # ETF列表（63只，可自行替换）
    }


# 因子加工配置
class FactorConfig:
    batch_size = 1000  # 批处理大小
    max_workers = 4    # 最大工作线程数


# 评分卡建模配置
class ScorecardConfig:
    test_size = 0.3            # 测试集比例
    random_seed = 42            # 随机种子
    C = 100.0                   # 正则化参数
    pdo = 50                    # 每翻倍赔率的分数变化
    score_0 = 600               # 基础分数
    odds_0 = 50                 # 基础赔率

    # 仓位配置
    up_ratio_threshold = 0.5    # 上涨比例阈值
    high_position_ratio = 0.8   # 高仓位比例
    low_position_ratio = 0.1    # 低仓位比例
    min_holding_days = 5        # 最小持仓天数


# 回测配置
class BacktestConfig:
    initial_capital = 1000000   # 初始资金
    top_ratio = 0.01             # 选择前N%的ETF
    transaction_cost = 0.0001   # 交易成本
    up_ratio_threshold = 0.5    # 上涨比例阈值
    high_position_ratio = 0.8   # 高仓位比例
    low_position_ratio = 0.1    # 低仓位比例
    min_holding_days = 5        # 最小持仓天数
    compare_indices = ['510300.SH', '510500.SH', '510050.SH']  # 对比指数


# 数据存储配置
class DataStorageConfig:
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


# QMT数据中心配置
class QMTConfig:
    token = QMT_TOKEN                       # ← 从环境变量 QMT_TOKEN 读取
    quote_time_mode = True
    data_home_dir = os.environ.get('QMT_DATA_DIR', r'C:\qmtdata')   # 设定自己的数据存储目录
    # 行情中转地址：请替换为你所用券商/数据服务提供方的地址
    addr_list = ['127.0.0.1:55310']


# QMT交易端配置（当前项目未实现下单逻辑，仅预留）
class QMTTraderConfig:
    path = os.environ.get('QMT_TRADER_PATH', r'C:\你的券商QMT交易端\userdata_mini')
    account_id = os.environ.get('QMT_ACCOUNT_ID', '')   # ← 切勿硬编码真实账户


# ============================================================
# 数据源配置（方案 A：双数据源并存，Tushare 优先 + QMT 可选）
# ============================================================
# provider 可选值：
#   'tushare' -> 强制使用 Tushare（推荐，跨平台，无需 QMT 环境）
#   'qmt'     -> 强制使用 QMT（需本机 QMT 环境，支持分钟级）
#   'csv'     -> 只读本地 data/etf_hist_data.csv
#   'auto'    -> 依次尝试 tushare -> qmt -> csv，取第一个可用的
#
# 切换方式：改 DataSourceConfig.provider 一行即可，主流程代码无需改动。
# 详见 data_provider.py 与 可行性分析-Tushare替代方案.md
# ============================================================
class TushareConfig:
    # 优先读环境变量 TUSHARE_TOKEN，未设置则用下方硬编码值
    token = TUSHARE_TOKEN or '在此填写你的 Tushare token'
    sleep = 0.12      # 频率限制保护：每次接口调用后的休眠秒数
    adjust = 'qfq'    # 复权方式：'qfq' 前复权（与 QMT front 对齐）| 'hfq' | ''


class DataSourceConfig:
    provider = 'auto'         # 'tushare' | 'qmt' | 'csv' | 'auto'
    force_refresh = False     # True 则忽略 pickle 缓存，强制重新拉取

