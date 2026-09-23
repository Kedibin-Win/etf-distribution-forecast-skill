#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF期权对冲分析脚本 - 计算行业ETF与宽基ETF的相关性及对冲比例
用于强势ETF轮动策略的系统性风险对冲
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import QMTConfig, DataSourceConfig, TushareConfig

# ---- 统一数据源（方案 A：Tushare 优先 + QMT 可选 + CSV 兜底）----
from data_provider import get_provider

try:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="xtquant")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="xtquant")
except ImportError:
    pass

# 兼容旧变量名
QMT_AVAILABLE = False
xtdata = None
xtdc = None


ETF_CATEGORIES = {
    '科技': ['半导体', '芯片', '人工智能', 'AI', '通信', '5G', '软件', '计算机', '大数据', '云计算', '信创', '半导体设备'],
    '医药': ['医疗', '医药', '创新药', '生物医药', '生物科技', '中药'],
    '消费': ['酒', '消费', '食品饮料', '家电', '旅游'],
    '金融': ['券商', '证券', '银行', '金融', '保险', '金融科技'],
    '周期': ['煤炭', '钢铁', '有色金属', '化工', '石化', '稀土'],
    '新能源': ['光伏', '新能源车', '新能源', '电池', '锂电池', '绿电'],
    '军工': ['军工', '国防'],
    '地产': ['房地产'],
    '农业': ['农业', '养殖', '粮食'],
    '其他': ['传媒', '游戏', '电子', '消费电子', '高端装备', '建材', '电力', '电网', '卫星']
}


def load_etf_list(json_path):
    """从JSON文件加载ETF列表"""
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            etf_dict = json.load(f)
        
        etf_list = []
        for code, name in etf_dict.items():
            category = '其他'
            for cat_name, keywords in ETF_CATEGORIES.items():
                if any(kw in name for kw in keywords):
                    category = cat_name
                    break
            
            etf_list.append({
                'code': code,
                'name': name,
                'category': category
            })
        
        print(f"从JSON加载ETF列表: {len(etf_list)}只")
        return etf_list
    else:
        print(f"ETF.json文件不存在: {json_path}")
        return []


# 宽基ETF（用于对冲的标的）
BROAD_ETFs = [
    {'code': '510300.SH', 'name': '沪深300ETF', 'option_available': True},
    {'code': '510050.SH', 'name': '上证50ETF', 'option_available': True},
    {'code': '510500.SH', 'name': '中证500ETF', 'option_available': True},
    {'code': '588000.SH', 'name': '科创50ETF', 'option_available': True},
    {'code': '159915.SZ', 'name': '创业板ETF', 'option_available': True},
]


class ETFHedgeAnalyzer:
    """ETF对冲分析器"""
    
    def __init__(self, sector_etfs=None, broad_etfs=None, start_date=None, end_date=None):
        self.sector_etfs = sector_etfs or []
        self.broad_etfs = broad_etfs or BROAD_ETFs
        self.start_date = start_date or '2025-01-01'
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.frequency = '1d'
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'hedge_data')
        self.cache_dir = os.path.join(self.base_dir, 'cache', 'hedge_data')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.force_refresh = False
        self.provider = None
    
    def _get_provider(self):
        """延迟获取数据源"""
        if self.provider is None:
            self.provider = get_provider(
                force_refresh=self.force_refresh,
                cache_dir=self.cache_dir,
            )
            print(f"   数据源: {self.provider.name}")
        return self.provider

    def _download_etf_data(self, etf_codes):
        """下载ETF OHLCV数据（统一走 data_provider）"""
        if not etf_codes:
            return {}

        provider = self._get_provider()
        data_dict = provider.get_ohlcv(
            etf_codes,
            self.start_date,
            self.end_date,
            frequency=self.frequency,
        )

        if not data_dict:
            print("数据源未返回数据，尝试本地CSV兜底...")
            data_dict = self._load_local_etf_data(etf_codes)

        return data_dict

    def _download_option_data(self, undl_code, dedate=None, opttype=''):
        """下载宽基ETF期权数据（统一走 data_provider）"""
        dedate = dedate or datetime.now().strftime('%Y%m')

        provider = self._get_provider()
        if not getattr(provider, 'supports_option', False):
            print(f"   数据源 {provider.name} 不支持期权数据，跳过 {undl_code}")
            return {}

        try:
            codes = provider.get_option_contracts(undl_code, dedate=dedate, opttype=opttype)
        except Exception as e:
            print(f"   获取期权合约失败: {e}")
            return {}

        print(f"   获取到期{dedate}的{undl_code}期权合约: {len(codes)}只")
        if not codes:
            return {}

        try:
            data_dict = provider.get_option_ohlcv(
                codes[:30], self.start_date, self.end_date, self.frequency
            )
        except Exception as e:
            print(f"   下载期权行情失败: {e}")
            return {}

        print(f"   期权数据下载完成: {len(data_dict)} 只")
        return data_dict

    def _get_cache_path(self, code):
        cache_key = f"{code}_{self.frequency}_{self.start_date}_{self.end_date}"
        cache_key = cache_key.replace('/', '_').replace('\\', '_')
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _load_cached_data(self, code):
        cache_path = self._get_cache_path(code)
        if os.path.exists(cache_path):
            try:
                df = pd.read_pickle(cache_path)
                return df
            except Exception:
                return None
        return None
    
    def _save_cached_data(self, code, df):
        cache_path = self._get_cache_path(code)
        try:
            df.to_pickle(cache_path)
            return True
        except Exception:
            return False
    
    def _load_local_etf_data(self, etf_codes):
        """从本地CSV加载ETF数据"""
        data_dict = {}
        for code in etf_codes:
            csv_path = os.path.join(self.data_dir, f'{code.replace(".", "_")}.csv')
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)
                    if len(df) > 1:
                        data_dict[code] = df
                except Exception:
                    continue
        
        print(f"从本地CSV加载完成: {len(data_dict)} 只ETF")
        return data_dict
    
    def _compute_returns(self, data_dict):
        """计算ETF日收益率"""
        returns_dict = {}
        for code, df in data_dict.items():
            if 'close' in df.columns:
                returns = pd.DataFrame()
                returns['date'] = df['date']
                returns['close'] = df['close']
                returns['return'] = df['close'].pct_change().fillna(0)
                returns_dict[code] = returns
        return returns_dict
    
    def _compute_correlation_matrix(self, returns_dict):
        """计算ETF之间的相关系数矩阵"""
        all_dates = set()
        for code, df in returns_dict.items():
            all_dates.update(df['date'].tolist())
        
        sorted_dates = sorted(list(all_dates))
        
        return_df = pd.DataFrame(index=sorted_dates)
        for code, df in returns_dict.items():
            df = df.set_index('date')
            df = df.reindex(sorted_dates)
            return_df[code] = df['return'].fillna(0)
        
        corr_matrix = return_df.corr()
        return corr_matrix, return_df
    
    def _find_atm_option(self, option_data, undl_price):
        """找到平值期权"""
        atm_option = None
        min_diff = float('inf')
        
        for code, df in option_data.items():
            if len(df) > 0:
                latest_price = df['close'].iloc[-1]
                diff = abs(latest_price - undl_price * 0.01)
                if diff < min_diff:
                    min_diff = diff
                    atm_option = code
        
        return atm_option
    
    def _calculate_hedge_ratio(self, sector_return, broad_return, window=60):
        """计算对冲比例（最小二乘法回归）"""
        merged = pd.DataFrame({'sector': sector_return, 'broad': broad_return}).dropna()
        
        if len(merged) < window:
            window = len(merged)
        
        rolling_beta = []
        for i in range(window, len(merged) + 1):
            window_data = merged.iloc[i - window:i]
            if window_data['broad'].std() > 0:
                beta = window_data['sector'].cov(window_data['broad']) / window_data['broad'].var()
            else:
                beta = 0
            rolling_beta.append(beta)
        
        avg_beta = np.mean(rolling_beta) if rolling_beta else 0
        recent_beta = rolling_beta[-1] if rolling_beta else 0
        std_beta = np.std(rolling_beta) if len(rolling_beta) > 1 else 0
        
        return {
            'avg_beta': avg_beta,
            'recent_beta': recent_beta,
            'std_beta': std_beta,
            'correlation': merged['sector'].corr(merged['broad']),
            'hedge_ratio': abs(avg_beta)
        }
    
    def _find_best_hedge_instrument(self, sector_returns, broad_returns, sector_code):
        """为行业ETF找到最佳对冲工具"""
        best_result = None
        max_corr = -1
        
        sector_return = sector_returns[sector_code]['return']
        
        for broad in self.broad_etfs:
            broad_code = broad['code']
            if broad_code not in broad_returns:
                continue
            
            broad_return = broad_returns[broad_code]['return']
            hedge_info = self._calculate_hedge_ratio(sector_return, broad_return)
            
            if abs(hedge_info['correlation']) > max_corr:
                max_corr = abs(hedge_info['correlation'])
                best_result = {
                    'sector_code': sector_code,
                    'sector_name': next(s['name'] for s in self.sector_etfs if s['code'] == sector_code),
                    'sector_category': next(s['category'] for s in self.sector_etfs if s['code'] == sector_code),
                    'broad_code': broad_code,
                    'broad_name': broad['name'],
                    'option_available': broad.get('option_available', False),
                    **hedge_info
                }
        
        if best_result:
            if best_result['option_available']:
                if abs(best_result['correlation']) > 0.7:
                    best_result['recommendation'] = '强烈推荐对冲'
                elif abs(best_result['correlation']) > 0.4:
                    best_result['recommendation'] = '推荐对冲'
                else:
                    best_result['recommendation'] = '相关性较低'
            else:
                best_result['recommendation'] = '标的无期权'
        
        return best_result
    
    def analyze(self, save_results=True):
        """执行对冲分析"""
        print("=" * 60)
        print("ETF期权对冲分析")
        print("=" * 60)
        print(f"  时间范围: {self.start_date} 到 {self.end_date}")
        print(f"  行业ETF数量: {len(self.sector_etfs)}")
        print(f"  宽基ETF数量: {len(self.broad_etfs)}")
        print("=" * 60)
        
        if not self.sector_etfs:
            print("行业ETF列表为空，程序退出")
            return False
        
        print("\n1. 获取行业ETF数据...")
        sector_codes = [etf['code'] for etf in self.sector_etfs]
        sector_data = self._download_etf_data(sector_codes)
        
        print("\n2. 获取宽基ETF数据...")
        broad_codes = [etf['code'] for etf in self.broad_etfs]
        broad_data = self._download_etf_data(broad_codes)
        
        if not sector_data or not broad_data:
            print("ETF数据获取失败，程序退出")
            return False
        
        print("\n3. 计算收益率...")
        sector_returns = self._compute_returns(sector_data)
        broad_returns = self._compute_returns(broad_data)
        
        print("\n4. 计算相关系数矩阵...")
        all_returns = {**sector_returns, **broad_returns}
        corr_matrix, return_df = self._compute_correlation_matrix(all_returns)
        
        print("\n5. 计算最佳对冲工具...")
        hedge_results = []
        for sector in self.sector_etfs:
            sector_code = sector['code']
            if sector_code in sector_returns:
                result = self._find_best_hedge_instrument(sector_returns, broad_returns, sector_code)
                if result:
                    hedge_results.append(result)
        
        print("\n6. 获取宽基ETF期权数据...")
        option_info = []
        for broad in self.broad_etfs:
            if broad.get('option_available', False):
                undl_price = broad_data[broad['code']]['close'].iloc[-1] if broad['code'] in broad_data else 0
                
                call_options = self._download_option_data(broad['code'], opttype='CALL')
                put_options = self._download_option_data(broad['code'], opttype='PUT')
                
                atm_call = self._find_atm_option(call_options, undl_price)
                atm_put = self._find_atm_option(put_options, undl_price)
                
                option_info.append({
                    'undl_code': broad['code'],
                    'undl_name': broad['name'],
                    'undl_price': undl_price,
                    'call_count': len(call_options),
                    'put_count': len(put_options),
                    'atm_call': atm_call,
                    'atm_put': atm_put
                })
        
        print("\n7. 生成分析报告...")
        if save_results:
            self._save_results(hedge_results, corr_matrix, option_info, return_df)
        
        self._print_report(hedge_results, corr_matrix, option_info)
        
        print("\n" + "=" * 60)
        print("ETF期权对冲分析完成!")
        print("=" * 60)
        
        return {
            'hedge_results': hedge_results,
            'corr_matrix': corr_matrix,
            'option_info': option_info,
            'return_df': return_df
        }
    
    def _save_results(self, hedge_results, corr_matrix, option_info, return_df):
        """保存分析结果"""
        hedge_df = pd.DataFrame(hedge_results)
        hedge_csv_path = os.path.join(self.data_dir, 'hedge_results.csv')
        hedge_df.to_csv(hedge_csv_path, index=False, encoding='utf-8-sig')
        print(f"对冲结果已保存到: {hedge_csv_path}")
        
        corr_csv_path = os.path.join(self.data_dir, 'correlation_matrix.csv')
        corr_matrix.to_csv(corr_csv_path, encoding='utf-8-sig')
        print(f"相关系数矩阵已保存到: {corr_csv_path}")
        
        option_df = pd.DataFrame(option_info)
        option_csv_path = os.path.join(self.data_dir, 'option_info.csv')
        option_df.to_csv(option_csv_path, index=False, encoding='utf-8-sig')
        print(f"期权信息已保存到: {option_csv_path}")
        
        return_csv_path = os.path.join(self.data_dir, 'etf_returns.csv')
        return_df.to_csv(return_csv_path, encoding='utf-8-sig')
        print(f"收益率数据已保存到: {return_csv_path}")
    
    def _print_report(self, hedge_results, corr_matrix, option_info):
        """打印分析报告"""
        print("\n" + "=" * 60)
        print("一、ETF相关系数矩阵（与宽基ETF）")
        print("=" * 60)
        
        broad_codes = [b['code'] for b in self.broad_etfs]
        sector_codes = [s['code'] for s in self.sector_etfs]
        
        corr_subset = corr_matrix.loc[sector_codes, broad_codes]
        print(corr_subset)
        
        print("\n" + "=" * 60)
        print("二、最佳对冲工具分析")
        print("=" * 60)
        
        hedge_df = pd.DataFrame(hedge_results)
        hedge_df = hedge_df.sort_values('correlation', key=lambda x: x.abs(), ascending=False)
        print(hedge_df[['sector_name', 'sector_category', 'broad_name', 
                       'correlation', 'avg_beta', 'hedge_ratio', 'recommendation']])
        
        print("\n" + "=" * 60)
        print("三、宽基ETF期权信息")
        print("=" * 60)
        
        for info in option_info:
            print(f"\n{info['undl_name']} ({info['undl_code']}):")
            print(f"  当前价格: {info['undl_price']:.2f}元")
            print(f"  认购期权数量: {info['call_count']}只")
            print(f"  认沽期权数量: {info['put_count']}只")
            print(f"  平值认购: {info['atm_call']}")
            print(f"  平值认沽: {info['atm_put']}")
        
        print("\n" + "=" * 60)
        print("四、对冲策略建议")
        print("=" * 60)
        print("""
对冲策略说明:
1. Beta系数: 表示行业ETF相对于宽基ETF的敏感度
   - Beta > 1: 行业ETF波动大于大盘
   - Beta < 1: 行业ETF波动小于大盘
   - Beta < 0: 行业ETF与大盘反向
        
2. 对冲仓位计算:
   对冲仓位 = |Beta| × 行业ETF持仓金额 / 期权合约单位
   例: 持有100万半导体ETF，Beta=1.2，使用510300认沽期权对冲
       对冲仓位 = 1.2 × 100万 / (510300现价 × 合约乘数)
        
3. 期权选择建议:
   - 首选近月平值认沽期权进行Delta对冲
   - 当预计大盘大幅下跌时，可选择虚值认沽期权
   - 滚动对冲: 每月到期前切换到下一期合约
        
4. 相关性筛选:
   - 相关性 > 0.7: 强相关，强烈推荐对冲
   - 相关性 0.4-0.7: 中等相关，推荐对冲
   - 相关性 < 0.4: 弱相关，不建议对冲
""")


def main():
    parser = argparse.ArgumentParser(description='ETF期权对冲分析脚本')
    parser.add_argument('--start-date', type=str, default='2025-01-01', help='开始日期(YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='结束日期(YYYY-MM-DD)')
    parser.add_argument('--force-refresh', action='store_true', help='强制刷新缓存')
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    etf_json_path = os.path.join(base_dir, 'ETF.json')
    sector_etfs = load_etf_list(etf_json_path)
    
    analyzer = ETFHedgeAnalyzer(
        sector_etfs=sector_etfs,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if args.force_refresh:
        analyzer.force_refresh = True
        print("⚠️ 强制刷新模式: 将重新下载所有数据")
    
    result = analyzer.analyze()
    
    if result:
        print("\n✅ ETF期权对冲分析完成")
        print(f"结果已保存到: {analyzer.data_dir}")
        sys.exit(0)
    else:
        print("\n❌ ETF期权对冲分析失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
