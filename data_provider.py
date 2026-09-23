#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一数据源抽象层（方案 A：双数据源并存）
=========================================
Tushare（优先） + QMT/xtquant（可选） + 本地CSV（兜底）

设计目标
--------
1. **不改动主流程逻辑**：训练 / 预测 / 回测 / 可视化的代码路径完全不变，
   只把「取数」这一层抽出来，换数据源 = 换一个 Provider。
2. **向后兼容**：config 里 provider 留空或设为 'qmt' 时，行为与改造前一致。
3. **统一返回结构**：所有 Provider 返回同构数据，调用方无需关心数据来自哪里。

统一接口契约
------------
所有 Provider 均实现以下方法（返回结构完全一致）：

    get_ohlcv(symbols, start_date, end_date, frequency='1d', adjust='qfq')
        -> Dict[ts_code, pd.DataFrame]
           DataFrame 列: ['date'(datetime64), 'open', 'high', 'low', 'close', 'volume']

    get_instrument_names(symbols) -> Dict[ts_code, str]        # 标的中文名

    get_option_contracts(undl_code, dedate=None, opttype='') -> List[ts_code]

    get_option_ohlcv(codes, start_date, end_date, frequency='1d') -> Dict[ts_code, pd.DataFrame]

    get_stock_list_in_sector(sector) -> List[ts_code]

选择方式
--------
    from config import DataSourceConfig
    DataSourceConfig.provider = 'tushare'   # 'tushare' | 'qmt' | 'csv' | 'auto'
    # 'auto' = 依次尝试 tushare -> qmt -> csv，取第一个可用的

    from data_provider import get_provider
    provider = get_provider()
    data = provider.get_ohlcv(symbols, '2016-01-01', '2026-06-24')
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 常量
# ============================================================

OHLCV_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']

# 分钟频率集合（Tushare 需单独开通权限，与积分无关）
MINUTE_FREQUENCIES = {'1m', '5m', '15m', '30m', '60m', '1h'}

TUSHARE_FREQ_MAP = {
    '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
    '60m': '60min', '1h': '60min',
}


# ============================================================
# 基类
# ============================================================

class BaseProvider:
    """数据源基类 —— 定义统一接口契约"""

    name = 'base'
    supports_minute = False      # 是否支持分钟级
    supports_option = False      # 是否支持期权数据

    def __init__(self, cache_dir=None, force_refresh=False):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'cache', 'market_data'
        )
        self.force_refresh = force_refresh
        os.makedirs(self.cache_dir, exist_ok=True)
        self._name_cache = {}
        self._contract_cache = {}

    # ---------- 缓存工具 ----------

    @staticmethod
    def _norm_date(d) -> str:
        """统一成 YYYYMMDD 字符串"""
        if d is None:
            return ''
        s = str(d).replace('-', '').replace('/', '').replace(' ', '')[:8]
        return s

    def _cache_path(self, key: str) -> str:
        safe = key.replace('/', '_').replace('\\', '_').replace(':', '_')
        return os.path.join(self.cache_dir, f"{safe}.pkl")

    def _load_cache(self, key: str):
        if self.force_refresh:
            return None
        p = self._cache_path(key)
        if os.path.exists(p):
            try:
                return pd.read_pickle(p)
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, df):
        try:
            df.to_pickle(self._cache_path(key))
        except Exception:
            pass

    # ---------- 统一接口（子类实现） ----------

    def get_ohlcv(self, symbols, start_date, end_date, frequency='1d', adjust='qfq'):
        raise NotImplementedError

    def get_instrument_names(self, symbols):
        return {}

    def get_option_contracts(self, undl_code, dedate=None, opttype=''):
        return []

    def get_option_ohlcv(self, codes, start_date, end_date, frequency='1d'):
        return {}

    def get_stock_list_in_sector(self, sector):
        return []

    # ---------- 共享工具 ----------

    @staticmethod
    def _standardize(df: pd.DataFrame) -> pd.DataFrame:
        """把任意来源的 DataFrame 统一成标准 OHLCV 结构"""
        if df is None or len(df) == 0:
            return None
        df = df.copy()
        if 'date' not in df.columns:
            return None
        df['date'] = pd.to_datetime(df['date'])
        for c in ['open', 'high', 'low', 'close', 'volume']:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df[OHLCV_COLUMNS].dropna(subset=['close'])
        df = df.sort_values('date').reset_index(drop=True)
        return df if len(df) > 1 else None


# ============================================================
# Tushare Provider（优先）
# ============================================================

class TushareProvider(BaseProvider):
    """
    基于 Tushare Pro 的数据源。

    接口映射：
        ETF 日线（前复权） → ts.pro_bar(asset='FD', adj='qfq')
        ETF 日线（不复权） → pro.fund_daily
        复权因子          → pro.fund_adj
        标的中文名        → pro.fund_basic(market='E')
        A股列表           → pro.stock_basic
        期权合约列表      → pro.opt_basic（opt_code = 'OP<标的代码>'）
        期权日线          → pro.opt_daily
    """

    name = 'tushare'
    supports_minute = False   # 分钟行情需单独开权限，非积分可解决
    supports_option = True

    def __init__(self, token=None, cache_dir=None, force_refresh=False, sleep=0.12):
        super().__init__(cache_dir=cache_dir, force_refresh=force_refresh)
        self.sleep = sleep
        try:
            import tushare as ts
        except ImportError:
            raise ImportError(
                "未安装 tushare，请执行:  pip install tushare\n"
                "（Windows 用户也可用: uv add tushare）"
            )
        self.ts = ts
        self.token = token or os.environ.get('TUSHARE_TOKEN', '')
        if not self.token:
            raise ValueError(
                "Tushare token 未配置。请在 config.TushareConfig.token 填写，"
                "或设置环境变量 TUSHARE_TOKEN。"
            )
        ts.set_token(self.token)
        self.pro = ts.pro_api(self.token)
        self._verify()

    def _verify(self):
        try:
            df = self.pro.trade_cal(exchange='SSE', start_date='20260101', end_date='20260105')
            if df is None or len(df) == 0:
                raise RuntimeError("trade_cal 返回空")
        except Exception as e:
            raise ConnectionError(f"Tushare token 校验失败: {e}")

    def _retry(self, fn, *args, **kwargs):
        """带退避重试的接口调用"""
        last_err = None
        for attempt in range(3):
            try:
                r = fn(*args, **kwargs)
                time.sleep(self.sleep)
                return r
            except Exception as e:
                last_err = e
                time.sleep(0.6 * (attempt + 1))
        print(f"   [Tushare] 调用失败: {last_err}")
        return None

    # ---------------- OHLCV ----------------

    def get_ohlcv(self, symbols, start_date, end_date, frequency='1d', adjust='qfq'):
        if frequency in MINUTE_FREQUENCIES:
            print(f"\n[Tushare] ⚠️ 分钟频率 '{frequency}' 需单独开通权限（与积分无关）。")
            print("          当前仅支持日线及以上频率，已跳过。")
            print("          如需分钟级，请在 QMT 环境运行或向 Tushare 申请分钟数据权限。")
            return {}

        s, e = self._norm_date(start_date), self._norm_date(end_date)
        out, hit, miss, fail = {}, 0, 0, 0
        total = len(symbols)

        for i, sym in enumerate(symbols, 1):
            key = f"ts_v3_{sym}_{frequency}_{adjust}_{s}_{e}"
            df = self._load_cache(key)
            if df is not None:
                out[sym] = df
                hit += 1
            else:
                df = self._fetch_one(sym, s, e, adjust)
                if df is not None:
                    out[sym] = df
                    self._save_cache(key, df)
                    miss += 1
                else:
                    fail += 1
            if i % 10 == 0 or i == total:
                print(f"   已处理: {i}/{total}, 缓存: {hit}, 下载: {miss}, 失败: {fail}")

        print(f"数据获取完成: 缓存 {hit} 只, 下载 {miss} 只, 失败 {fail} 只")
        return out

    def _fetch_one(self, sym, start_date, end_date, adjust='qfq'):
        """
        获取单只 ETF 的 OHLCV。

        ⚠️ 重要实测结论：`ts.pro_bar(asset='FD', adj='qfq')` 对 ETF **不做复权**
        （输出与 fund_daily 不复权完全一致），遇到 ETF 份额拆分/合并日会产生
        假暴跌（实测 510230.SH 2020-08-17 为 -78.8%、159928.SZ 2021-06-25 为 -74.5%）。
        因此这里**统一用 fund_daily（原始） + fund_adj（复权因子）手工换算**。

        前复权公式：qfq = 原价 × 当日因子 ÷ 最新因子
        后复权公式：hfq = 原价 × 当日因子
        """
        raw = self._retry(self.pro.fund_daily, ts_code=sym,
                          start_date=start_date, end_date=end_date)
        if raw is None or len(raw) == 0:
            return None

        df = raw.rename(columns={'trade_date': 'date', 'vol': 'volume'})
        df = df.sort_values('date').reset_index(drop=True)

        if adjust in ('qfq', 'hfq'):
            adj = self._retry(self.pro.fund_adj, ts_code=sym,
                              start_date=start_date, end_date=end_date)
            if adj is not None and len(adj):
                adj = adj.rename(columns={'trade_date': 'date'})[['date', 'adj_factor']]
                adj = adj.sort_values('date').drop_duplicates('date')
                df = df.merge(adj, on='date', how='left')
                df['adj_factor'] = df['adj_factor'].ffill().bfill().fillna(1.0)

                factor = df['adj_factor'].astype(float)
                if adjust == 'qfq':
                    latest = factor.iloc[-1] if len(factor) else 1.0
                    factor = factor / (latest if latest else 1.0)

                # 价格 ÷ 复权
                for c in ['open', 'high', 'low', 'close']:
                    df[c] = df[c] * factor

                # 成交量同口径复权：份额拆分/合并会让成交量（份数）跳变，
                # 与价格因子反向缩放后才是可比的"交易活跃度"。
                # 例：159928.SZ 2021-06-25 份额拆分，价格 ÷3.9、成交量 ×5.6，
                #    不做此处理则 volume_change 达 790 倍（原数据仅约 1.4 倍）。
                df['volume'] = df['volume'] / factor.replace(0, np.nan)

                # 复权后价格不可为负/零
                df = df[(df['close'] > 0)]
            else:
                # 无复权因子（无分红拆分记录）—— 原始价即复权价
                pass

        return self._standardize(df)


    # ---------------- 标的名称 ----------------

    def get_instrument_names(self, symbols):
        if self._name_cache:
            return {s: self._name_cache[s] for s in symbols if s in self._name_cache}

        names = {}
        fb = self._retry(self.pro.fund_basic, market='E')
        if fb is not None and len(fb):
            names.update(dict(zip(fb['ts_code'], fb['name'])))
            print(f"   通过Tushare获取到 {len(names)} 个基金名称")

        # 股票（sector 模式兜底）
        sb = self._retry(self.pro.stock_basic, exchange='', list_status='L')
        if sb is not None and len(sb):
            names.update(dict(zip(sb['ts_code'], sb['name'])))

        self._name_cache = names
        return {s: names.get(s, s) for s in symbols}

    # ---------------- 板块成分 ----------------

    def get_stock_list_in_sector(self, sector):
        if sector in ('沪深A股', 'A股', '全部A股'):
            sb = self._retry(self.pro.stock_basic, exchange='', list_status='L')
            if sb is not None and len(sb):
                return sb['ts_code'].tolist()
            return []
        # 其他板块名尝试当指数处理
        try:
            mem = self._retry(self.pro.index_member, index_code=sector)
            if mem is not None and len(mem):
                return mem['con_code'].tolist()
        except Exception:
            pass
        return []

    # ---------------- 期权 ----------------

    def _load_option_basic(self, exchange):
        key = f"_optbasic_{exchange}"
        if key in self._contract_cache:
            return self._contract_cache[key]
        ob = self._retry(self.pro.opt_basic, exchange=exchange)
        if ob is None:
            ob = pd.DataFrame()
        self._contract_cache[key] = ob
        return ob

    def get_option_contracts(self, undl_code, dedate=None, opttype=''):
        """
        按标的代码取期权合约。
        Tushare 的 opt_basic.opt_code 形如 'OP510300.SH'，即 'OP' + 标的代码。
        """
        code = undl_code.split('.')[0]
        exchange = 'SSE' if undl_code.upper().endswith('.SH') else 'SZSE'
        ob = self._load_option_basic(exchange)
        if ob is None or len(ob) == 0:
            print(f"   [Tushare] opt_basic({exchange}) 无数据")
            return []

        target = f"OP{code}.{exchange}"
        sub = ob[ob['opt_code'].astype(str).str.upper() == target]
        if len(sub) == 0:
            # 退化匹配：名称含标的前缀
            sub = ob[ob['opt_code'].astype(str).str.contains(code, na=False)]

        if opttype:
            want = 'C' if str(opttype).upper().startswith('C') else 'P'
            sub = sub[sub['call_put'].astype(str).str.upper() == want]

        if dedate:
            d = str(dedate)
            if len(d) >= 6:
                sm = d[:6]
                sub = sub[sub['s_month'].astype(str).str.startswith(sm)]

        return sub['ts_code'].tolist()

    def get_option_ohlcv(self, codes, start_date, end_date, frequency='1d'):
        s, e = self._norm_date(start_date), self._norm_date(end_date)
        out = {}
        for code in codes:
            key = f"tsopt_{code}_{s}_{e}"
            df = self._load_cache(key)
            if df is None:
                raw = self._retry(self.pro.opt_daily, ts_code=code,
                                  start_date=s, end_date=e)
                if raw is None or len(raw) == 0:
                    continue
                raw = raw.rename(columns={'trade_date': 'date', 'vol': 'volume'})
                df = self._standardize(raw)
                if df is not None:
                    self._save_cache(key, df)
            if df is not None:
                out[code] = df
        return out


# ============================================================
# QMT Provider（可选，保留原有能力）
# ============================================================

class QMTProvider(BaseProvider):
    """
    基于 QMT / xtquant 的数据源（原项目逻辑，予以保留）。

    优势：支持分钟级、支持实时期点、本地数据中心无网络限制。
    代价：需本机安装 QMT 环境，Windows 专属。
    """

    name = 'qmt'
    supports_minute = True
    supports_option = True

    def __init__(self, cache_dir=None, force_refresh=False):
        super().__init__(cache_dir=cache_dir, force_refresh=force_refresh)
        self.xtdata = None
        self.available = False
        self._init_xt()

    def _init_xt(self):
        try:
            warnings.filterwarnings("ignore", category=UserWarning, module="xtquant")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="xtquant")
            from xtquant import xtdata
            from xtquant import xtdatacenter as xtdc
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from config import QMTConfig

            try:
                xtdc.set_token(QMTConfig.token)
                xtdc.set_quote_time_mode_v2(QMTConfig.quote_time_mode)
                xtdc.set_data_home_dir(QMTConfig.data_home_dir)
                xtdc.set_index_mirror_enabled(True)
                xtdc.set_future_realtime_mode(True)
                xtdc.init(False)
                xtdc.listen(port=58615)
                print("QMT数据中心连接成功")
            except Exception as e:
                print(f"QMT数据中心连接失败: {e}")
                self.xtdata = None
                self.available = False
                return

            self.xtdata = xtdata
            self.available = True
        except ImportError:
            print("QMT模块导入失败（未安装 xtquant），该数据源不可用")
            self.xtdata = None
            self.available = False

    @staticmethod
    def _to_date(ts):
        from datetime import datetime
        s = str(ts)
        if len(s) == 8:
            return datetime.strptime(s, '%Y%m%d')
        return datetime.fromtimestamp(ts / 1000)

    def get_ohlcv(self, symbols, start_date, end_date, frequency='1d', adjust='qfq'):
        if not self.available:
            return {}
        s, e = self._norm_date(start_date), self._norm_date(end_date)
        out, hit, miss, fail = {}, 0, 0, 0
        total = len(symbols)

        for i, sym in enumerate(symbols, 1):
            key = f"qmt_{sym}_{frequency}_{s}_{e}"
            df = self._load_cache(key)
            if df is not None:
                out[sym] = df
                hit += 1
                continue

            try:
                try:
                    self.xtdata.download_history_data(sym, frequency,
                                                      s, e)
                except Exception:
                    pass
                time.sleep(0.05)

                data = self.xtdata.get_market_data_ex(
                    field_list=["time", "open", "high", "low", "close", "volume"],
                    stock_list=[sym],
                    period=frequency,
                    start_time=s,
                    end_time=e,
                    count=-1
                )
                if sym in data and hasattr(data[sym], 'columns'):
                    raw = data[sym]
                    if all(c in raw.columns for c in ['time', 'open', 'high', 'low', 'close', 'volume']):
                        df = pd.DataFrame({
                            'date': [self._to_date(t) for t in raw['time'].tolist()],
                            'open': raw['open'].tolist(),
                            'high': raw['high'].tolist(),
                            'low': raw['low'].tolist(),
                            'close': raw['close'].tolist(),
                            'volume': raw['volume'].tolist(),
                        })
                        df = self._standardize(df)
                        if df is not None:
                            out[sym] = df
                            self._save_cache(key, df)
                            miss += 1
                        else:
                            fail += 1
                    else:
                        fail += 1
                else:
                    fail += 1
            except Exception:
                fail += 1

            if i % 10 == 0 or i == total:
                print(f"   已处理: {i}/{total}, 缓存: {hit}, 下载: {miss}, 失败: {fail}")

        print(f"数据获取完成: 缓存 {hit} 只, 下载 {miss} 只, 失败 {fail} 只")
        return out

    def get_instrument_names(self, symbols):
        if not self.available:
            return {}
        names = {}
        for sym in symbols:
            try:
                detail = self.xtdata.get_instrument_detail(sym)
                if detail and 'InstrumentName' in detail:
                    names[sym] = detail['InstrumentName']
            except Exception:
                pass
        print(f"   通过QMT获取到 {len(names)} 个标的名称")
        return names

    def get_option_contracts(self, undl_code, dedate=None, opttype=''):
        if not self.available:
            return []
        from datetime import datetime
        dedate = dedate or datetime.now().strftime('%Y%m')
        try:
            return self.xtdata.get_option_list(
                undl_code=undl_code, dedate=dedate,
                opttype=opttype, isavailavle=True
            ) or []
        except Exception as e:
            print(f"   [QMT] 获取期权合约失败: {e}")
            return []

    def get_option_ohlcv(self, codes, start_date, end_date, frequency='1d'):
        if not self.available:
            return {}
        s, e = self._norm_date(start_date), self._norm_date(end_date)
        out = {}
        for code in codes[:30]:
            key = f"qmtopt_{code}_{frequency}_{s}_{e}"
            df = self._load_cache(key)
            if df is None:
                try:
                    self.xtdata.download_history_data(code, frequency)
                    time.sleep(0.03)
                    data = self.xtdata.get_market_data_ex(
                        field_list=[], stock_list=[code], period=frequency,
                        start_time=s, end_time=e, count=-1,
                        dividend_type='front', fill_data=True
                    )
                    if code in data and hasattr(data[code], 'columns'):
                        raw = data[code]
                        d = raw.reset_index()
                        d = d.rename(columns={d.columns[0]: 'date', 'vol': 'volume'})
                        df = self._standardize(d)
                        if df is not None:
                            self._save_cache(key, df)
                except Exception:
                    continue
            if df is not None:
                out[code] = df
        return out

    def get_stock_list_in_sector(self, sector):
        if not self.available:
            return []
        try:
            return self.xtdata.get_stock_list_in_sector(sector) or []
        except Exception:
            return []


# ============================================================
# CSV Provider（兜底）
# ============================================================

class CSVProvider(BaseProvider):
    """
    本地 CSV 兜底数据源（无网络/无凭证时的最后防线）。

    期望格式（单文件宽表）：
        data/etf_hist_data.csv   列: symbol, timestamp, open, high, low, close, volume
    或按标的分文件：
        data/{code_with_underscore}.csv   列: date, open, high, low, close, volume
    """

    name = 'csv'
    supports_minute = False
    supports_option = False

    def __init__(self, data_dir=None, cache_dir=None, force_refresh=False):
        super().__init__(cache_dir=cache_dir, force_refresh=force_refresh)
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data'
        )
        os.makedirs(self.data_dir, exist_ok=True)

    def get_ohlcv(self, symbols, start_date, end_date, frequency='1d', adjust='qfq'):
        # 1) 优先尝试单文件宽表
        wide = os.path.join(self.data_dir, 'etf_hist_data.csv')
        if os.path.exists(wide):
            try:
                df = pd.read_csv(wide)
                if 'symbol' in df.columns and 'timestamp' in df.columns:
                    df = df.rename(columns={'timestamp': 'date'})
                    out = {}
                    for sym, g in df.groupby('symbol'):
                        if symbols and sym not in symbols:
                            continue
                        sdf = self._standardize(g)
                        if sdf is not None:
                            out[sym] = sdf
                    if out:
                        print(f"从本地CSV({os.path.basename(wide)})加载完成: {len(out)} 只")
                        return out
            except Exception as e:
                print(f"读取 {wide} 失败: {e}")

        # 2) 退化为分文件
        out = {}
        for sym in symbols:
            p = os.path.join(self.data_dir, f"{sym.replace('.', '_')}.csv")
            if os.path.exists(p):
                try:
                    sdf = self._standardize(pd.read_csv(p))
                    if sdf is not None:
                        out[sym] = sdf
                except Exception:
                    continue
        print(f"从本地CSV分文件加载完成: {len(out)} 只")
        return out


# ============================================================
# 工厂
# ============================================================

_PROVIDER_CACHE = {}


def get_provider(name=None, force_refresh=False, **kwargs):
    """
    获取数据源实例。

    name: 'tushare' | 'qmt' | 'csv' | 'auto' | None
          None / 'auto' -> 读 config.DataSourceConfig.provider
          'auto'        -> 依次尝试 tushare -> qmt -> csv
    """
    if name is None:
        try:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from config import DataSourceConfig
            name = getattr(DataSourceConfig, 'provider', 'auto')
        except Exception:
            name = 'auto'

    if name == 'auto':
        for candidate in ('tushare', 'qmt', 'csv'):
            p = _try_provider(candidate, force_refresh, **kwargs)
            if p is not None:
                print(f"✅ 数据源自动选择: {p.name}")
                return p
        print("⚠️ 所有数据源均不可用，已回落到 CSVProvider")
        return CSVProvider(**{k: v for k, v in kwargs.items()
                              if k in ('data_dir', 'cache_dir')})

    p = _try_provider(name, force_refresh, **kwargs)
    if p is None:
        print(f"⚠️ 数据源 '{name}' 不可用，已回落到 CSVProvider")
        return CSVProvider()
    return p


def _try_provider(name, force_refresh, **kwargs):
    name = str(name).lower()
    if name == 'tushare':
        try:
            from config import TushareConfig
            token = kwargs.pop('token', None) or TushareConfig.token
            return TushareProvider(token=token, force_refresh=force_refresh, **kwargs)
        except Exception as e:
            print(f"   [tushare] 不可用: {e}")
            return None
    if name == 'qmt':
        try:
            p = QMTProvider(force_refresh=force_refresh)
            return p if p.available else None
        except Exception as e:
            print(f"   [qmt] 不可用: {e}")
            return None
    if name == 'csv':
        try:
            return CSVProvider(**{k: v for k, v in kwargs.items()
                                  if k in ('data_dir', 'cache_dir')})
        except Exception as e:
            print(f"   [csv] 不可用: {e}")
            return None
    return None
