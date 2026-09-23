#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据生成脚本 - 从真实ETF数据生成OHLCV高维训练数据供main.py使用
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DataConfig, QMTConfig, DataSourceConfig, TushareConfig

# ---- 统一数据源（方案 A：Tushare 优先 + QMT 可选 + CSV 兜底）----
from data_provider import get_provider

try:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="xtquant")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="xtquant")
except ImportError:
    pass

# 兼容旧变量名（历史代码可能引用；实际取数已全部走 data_provider）
QMT_AVAILABLE = False
xtdata = None
xtdc = None


class TrainDataGenerator:
    """训练数据生成器 - OHLCV多特征版本"""
    
    FEATURES = ['open_return', 'high_return', 'low_return', 'close_return', 
                'volume_change', 'range_ratio', 'body_ratio', 'upper_shadow', 'lower_shadow',
                'ma5', 'ma10', 'ma20', 'ma60', 'std5', 'std20', 'rsi', 'macd', 'macd_signal',
                'correlation', 'market_rank']
    
    def __init__(self):
        self.start_date = DataConfig.start_date
        self.end_date = DataConfig.end_date
        self.frequency = DataConfig.frequency
        self.stock_pool = DataConfig.stock_pool
        
        self.input_frequency = DataConfig.input_frequency
        self.input_window = DataConfig.input_window
        self.target_frequency = DataConfig.target_frequency
        
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'market_data')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.force_refresh = False
        self.provider = None  # 延迟初始化（需等 force_refresh 设置完成）

    def _get_provider(self):
        """延迟获取数据源（保证 force_refresh 已生效）"""
        if self.provider is None:
            self.provider = get_provider(
                force_refresh=self.force_refresh,
                cache_dir=self.cache_dir,
            )
            print(f"   数据源: {self.provider.name}")
        return self.provider
    
    def _get_stock_pool(self):
        """获取股票池"""
        stock_pool_type = self.stock_pool.get('type', 'sector')
        symbols = []
        
        if stock_pool_type == 'sector':
            sectors = self.stock_pool.get('sectors', [])
            provider = self._get_provider()
            for sector in sectors:
                try:
                    sector_symbols = provider.get_stock_list_in_sector(sector)
                    symbols.extend(sector_symbols)
                    print(f"获取{sector}板块成分股: {len(sector_symbols)}只")
                except Exception as e:
                    print(f"获取{sector}板块成分股失败: {e}")
        elif stock_pool_type == 'custom':
            symbols = self.stock_pool.get('custom_symbols', [])
            print(f"使用自定义股票池: {len(symbols)}只")
        elif stock_pool_type == 'etf':
            symbols = self.stock_pool.get('etf_list', [])
            print(f"使用ETF列表: {len(symbols)}只")
        elif stock_pool_type == 'csv':
            csv_path = self.stock_pool.get('csv_path', '')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'symbol' in df.columns:
                    symbols = df['symbol'].tolist()
                elif 'code' in df.columns:
                    symbols = df['code'].tolist()
                print(f"从CSV文件加载股票池: {csv_path}, 共 {len(symbols)}只")
            else:
                print(f"CSV文件不存在: {csv_path}")
        
        # 保序去重（原为 list(set(symbols))，会打乱顺序且受 PYTHONHASHSEED 影响不可复现；
        # 而特征列顺序 = 股票池顺序，一旦错位会导致 predict.py 的标的名称映射整体错配）
        symbols = list(dict.fromkeys(symbols))
        print(f"最终股票池: {len(symbols)}只")
        return symbols

    
    def _download_data(self, symbols):
        """下载/加载OHLCV历史数据（统一走 data_provider 数据源层）"""
        provider = self._get_provider()

        print(f"   [数据源={provider.name}] 开始获取 {len(symbols)} 只标的行情 "
              f"({self.start_date} ~ {self.end_date}, {self.frequency})")

        data_dict = provider.get_ohlcv(
            symbols,
            self.start_date,
            self.end_date,
            frequency=self.frequency,
        )

        if not data_dict:
            print("数据源未返回有效数据，尝试本地CSV兜底...")
            data_dict = self._load_local_data()

        if data_dict:
            first = next(iter(data_dict.values()))
            print(f"   实际获取到 {len(data_dict)} 只，样本区间: "
                  f"{first['date'].iloc[0]} ~ {first['date'].iloc[-1]}")

        return data_dict

    def _load_local_data(self):
        """从本地CSV加载OHLCV数据"""
        csv_path = os.path.join(self.data_dir, 'etf_hist_data.csv')
        if not os.path.exists(csv_path):
            print(f"本地CSV文件不存在: {csv_path}")
            return {}
        
        print(f"从本地CSV文件加载数据: {csv_path}")
        df = pd.read_csv(csv_path)
        
        data_dict = {}
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            symbol_df['date'] = pd.to_datetime(symbol_df['timestamp'])
            symbol_df = symbol_df[['date', 'open', 'high', 'low', 'close', 'volume']].sort_values('date').reset_index(drop=True)
            if len(symbol_df) > 1:
                data_dict[symbol] = symbol_df
        
        print(f"从本地CSV加载完成: {len(data_dict)} 只ETF")
        return data_dict
    
    def _get_cache_path(self, symbol):
        cache_key = f"{symbol}_{self.frequency}_{self.start_date}_{self.end_date}"
        cache_key = cache_key.replace('/', '_').replace('\\', '_')
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _load_cached_data(self, symbol):
        cache_path = self._get_cache_path(symbol)
        if os.path.exists(cache_path):
            try:
                df = pd.read_pickle(cache_path)
                return df
            except Exception:
                return None
        return None
    
    def _save_cached_data(self, symbol, df):
        cache_path = self._get_cache_path(symbol)
        try:
            df.to_pickle(cache_path)
            return True
        except Exception:
            return False
    
    def _get_freq_minutes(self, freq):
        """将频率字符串转换为分钟数"""
        freq = freq.lower()
        if freq == '1m':
            return 1
        elif freq == '5m':
            return 5
        elif freq == '15m':
            return 15
        elif freq == '30m':
            return 30
        elif freq == '1h':
            return 60
        elif freq == '1d':
            return 1440
        elif freq == '1w':
            return 10080
        else:
            raise ValueError(f"未知频率: {freq}")
    
    def _aggregate_to_frequency(self, df, target_freq):
        """将数据聚合到目标频率"""
        if target_freq == self.frequency:
            return df
        
        minutes = self._get_freq_minutes(target_freq)
        if minutes == 1440:
            df = df.set_index('date')
            daily_df = pd.DataFrame()
            daily_df['open'] = df['open'].resample('D').first()
            daily_df['high'] = df['high'].resample('D').max()
            daily_df['low'] = df['low'].resample('D').min()
            daily_df['close'] = df['close'].resample('D').last()
            daily_df['volume'] = df['volume'].resample('D').sum()
            daily_df = daily_df.dropna()
            daily_df['date'] = daily_df.index
            daily_df = daily_df.reset_index(drop=True)
            return daily_df
        else:
            df = df.set_index('date')
            rule = f'{minutes}T'
            agg_df = pd.DataFrame()
            agg_df['open'] = df['open'].resample(rule).first()
            agg_df['high'] = df['high'].resample(rule).max()
            agg_df['low'] = df['low'].resample(rule).min()
            agg_df['close'] = df['close'].resample(rule).last()
            agg_df['volume'] = df['volume'].resample(rule).sum()
            agg_df = agg_df.dropna()
            agg_df['date'] = agg_df.index
            agg_df = agg_df.reset_index(drop=True)
            return agg_df
    
    def _compute_features(self, data_dict, frequency=None):
        """计算OHLCV特征（支持指定频率）"""
        if not data_dict:
            print("无数据可处理")
            return None, None
        
        target_freq = frequency or self.frequency
        
        agg_data_dict = {}
        for symbol, df in data_dict.items():
            agg_df = self._aggregate_to_frequency(df, target_freq)
            agg_data_dict[symbol] = agg_df
        
        all_dates = set()
        for symbol, df in agg_data_dict.items():
            all_dates.update(df['date'].tolist())
        
        sorted_dates = sorted(list(all_dates))
        print(f"数据时间范围 ({target_freq}): {sorted_dates[0].strftime('%Y-%m-%d')} 到 {sorted_dates[-1].strftime('%Y-%m-%d')}")
        
        features_dict = {}
        for symbol, df in agg_data_dict.items():
            df = df.set_index('date')
            df = df.reindex(sorted_dates)
            df = df.ffill()
            df = df.bfill()
            
            if len(df) > 1:
                prev_close = df['close'].shift(1)
                
                features = pd.DataFrame(index=df.index)
                features['open_return'] = df['open'] / prev_close - 1
                features['high_return'] = df['high'] / prev_close - 1
                features['low_return'] = df['low'] / prev_close - 1
                features['close_return'] = df['close'] / prev_close - 1
                
                prev_volume = df['volume'].shift(1)
                features['volume_change'] = df['volume'] / prev_volume - 1
                
                features['range_ratio'] = (df['high'] - df['low']) / prev_close
                features['body_ratio'] = np.abs(df['close'] - df['open']) / prev_close
                features['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / prev_close
                features['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / prev_close
                
                features['ma5'] = df['close'].rolling(5).mean() / prev_close - 1
                features['ma10'] = df['close'].rolling(10).mean() / prev_close - 1
                features['ma20'] = df['close'].rolling(20).mean() / prev_close - 1
                features['ma60'] = df['close'].rolling(60).mean() / prev_close - 1
                
                features['std5'] = df['close'].rolling(5).std() / prev_close
                features['std20'] = df['close'].rolling(20).std() / prev_close
                
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                rs = gain / loss.replace(0, 1e-8)
                features['rsi'] = 100 - (100 / (1 + rs))
                
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                features['macd'] = (ema12 - ema26) / prev_close
                features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
                
                features_dict[symbol] = features
        
        market_close_df = pd.DataFrame()
        for symbol, features in features_dict.items():
            market_close_df[symbol] = features['close_return']
        market_avg_return = market_close_df.mean(axis=1)
        
        all_dates = sorted_dates[1:]
        
        for symbol, features in features_dict.items():
            features = features.reindex(all_dates)
            
            features['correlation'] = features['close_return'].rolling(20).corr(market_avg_return).fillna(0)
            
            rank_pct = market_close_df.rank(axis=1, ascending=False).apply(lambda x: x / len(market_close_df.columns), axis=1)
            features['market_rank'] = rank_pct[symbol] if symbol in rank_pct.columns else 0.5
            
            features = features.ffill().bfill()
            features = features.fillna(0)
            
            features_dict[symbol] = features
        
        return features_dict, all_dates
    
    def _build_training_data(self, features_dict, dates):
        """构建高维训练数据 (OHLCV特征)"""
        if not features_dict:
            print("无特征数据")
            return None
        
        symbol_list = list(features_dict.keys())
        num_symbols = len(symbol_list)
        num_features = len(self.FEATURES)
        print(f"有效ETF数量: {num_symbols}")
        print(f"特征数量: {num_features}")
        print(f"特征列表: {self.FEATURES}")
        
        market_data = []
        for date in dates:
            day_features = []
            for symbol in symbol_list:
                row = features_dict[symbol].loc[date]
                valid_features = []
                for feat in self.FEATURES:
                    val = row.get(feat, np.nan)
                    if not np.isnan(val) and not np.isinf(val):
                        valid_features.append(val)
                    else:
                        valid_features.append(0.0)
                day_features.extend(valid_features)
            market_data.append(day_features)
        
        market_data = np.array(market_data)
        print(f"有效时间点数量: {len(dates)}")
        print(f"每时间点特征维度: {num_symbols} 只ETF × {num_features} 特征 = {num_symbols * num_features}")
        print(f"数据形状: {market_data.shape}")
        print(f"数据统计 - 均值: {market_data.mean():.6f}, 标准差: {market_data.std():.6f}")
        print(f"数据统计 - 最小值: {market_data.min():.6f}, 最大值: {market_data.max():.6f}")
        
        return market_data, dates
    
    def _build_cross_period_training_data(self, input_features, input_dates, target_features, target_dates):
        """构建跨周期训练数据：input_frequency窗口 → target_frequency标签"""
        if not input_features or not target_features:
            print("输入或目标特征数据为空")
            return None, None
        
        symbol_list = list(input_features.keys())
        num_symbols = len(symbol_list)
        num_features = len(self.FEATURES)
        input_dim = self.input_window * num_symbols * num_features
        output_dim = num_symbols * num_features
        
        print(f"\n跨周期训练数据构建:")
        print(f"  输入频率: {self.input_frequency}, 输入窗口: {self.input_window}")
        print(f"  目标频率: {self.target_frequency}")
        print(f"  输入维度: {input_dim}")
        print(f"  输出维度: {output_dim}")
        
        input_date_to_idx = {d: i for i, d in enumerate(input_dates)}
        
        input_market_data, _ = self._build_training_data(input_features, input_dates)
        target_market_data, _ = self._build_training_data(target_features, target_dates)
        
        X_data = []
        y_data = []
        valid_date_pairs = []
        
        for target_date in target_dates:
            target_date_start = target_date.replace(hour=0, minute=0, second=0)
            
            search_end = target_date
            search_start = search_end - pd.Timedelta(days=7)
            
            relevant_input_dates = [d for d in input_dates if search_start <= d < search_end]
            
            if len(relevant_input_dates) >= self.input_window:
                window_input_dates = relevant_input_dates[-self.input_window:]
                window_indices = [input_date_to_idx[d] for d in window_input_dates]
                
                x = input_market_data[window_indices].flatten()
                
                target_idx = target_dates.index(target_date)
                y = target_market_data[target_idx]
                
                X_data.append(x)
                y_data.append(y)
                valid_date_pairs.append((window_input_dates[0], window_input_dates[-1], target_date))
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        print(f"  跨周期样本数量: {len(X_data)}")
        print(f"  X数据形状: {X_data.shape}")
        print(f"  y数据形状: {y_data.shape}")
        
        return X_data, y_data, valid_date_pairs
    
    def _normalize_data(self, market_data):
        """标准化数据"""
        self.data_mean = np.mean(market_data, axis=0)
        self.data_std = np.std(market_data, axis=0)
        self.data_std[self.data_std < 1e-8] = 1e-8
        
        normalized_data = (market_data - self.data_mean) / self.data_std
        print(f"\n数据标准化 - 均值范围: [{self.data_mean.min():.6f}, {self.data_mean.max():.6f}]")
        print(f"数据标准化 - 标准差范围: [{self.data_std.min():.6f}, {self.data_std.max():.6f}]")
        
        return normalized_data
    
    def _save_training_data(self, market_data, dates):
        """保存训练数据（含标准化参数）"""
        data_save_path = os.path.join(self.output_dir, 'training_data.npy')
        dates_save_path = os.path.join(self.output_dir, 'training_dates.npy')
        mean_save_path = os.path.join(self.output_dir, 'data_mean.npy')
        std_save_path = os.path.join(self.output_dir, 'data_std.npy')
        
        normalized_data = self._normalize_data(market_data)
        
        np.save(data_save_path, normalized_data)
        
        date_strings = [d.strftime('%Y-%m-%d') for d in dates]
        np.save(dates_save_path, np.array(date_strings))
        
        np.save(mean_save_path, self.data_mean)
        np.save(std_save_path, self.data_std)
        
        print(f"\n训练数据已保存到: {data_save_path}")
        print(f"日期数据已保存到: {dates_save_path}")
        print(f"标准化参数已保存到: {mean_save_path}, {std_save_path}")
        print(f"数据形状: {normalized_data.shape}")
        print(f"日期范围: {dates[0].strftime('%Y-%m-%d')} 到 {dates[-1].strftime('%Y-%m-%d')}")
        
        return True
    
    def _save_cross_period_data(self, X_data, y_data, date_pairs):
        """保存跨周期训练数据（含标准化参数）"""
        X_save_path = os.path.join(self.output_dir, 'training_data.npy')
        y_save_path = os.path.join(self.output_dir, 'training_labels.npy')
        date_pairs_save_path = os.path.join(self.output_dir, 'training_date_pairs.npy')
        mean_save_path = os.path.join(self.output_dir, 'data_mean.npy')
        std_save_path = os.path.join(self.output_dir, 'data_std.npy')
        input_mean_save_path = os.path.join(self.output_dir, 'input_mean.npy')
        input_std_save_path = os.path.join(self.output_dir, 'input_std.npy')
        
        self.y_mean = np.mean(y_data, axis=0)
        self.y_std = np.std(y_data, axis=0)
        self.y_std[self.y_std < 1e-8] = 1e-8
        normalized_y = (y_data - self.y_mean) / self.y_std
        
        self.x_mean = np.mean(X_data, axis=0)
        self.x_std = np.std(X_data, axis=0)
        self.x_std[self.x_std < 1e-8] = 1e-8
        normalized_x = (X_data - self.x_mean) / self.x_std
        
        np.save(X_save_path, normalized_x)
        np.save(y_save_path, normalized_y)
        
        date_strings = [(d1.strftime('%Y-%m-%d'), 
                         d2.strftime('%Y-%m-%d'), 
                         d3.strftime('%Y-%m-%d')) 
                        for d1, d2, d3 in date_pairs]
        np.save(date_pairs_save_path, np.array(date_strings))
        
        np.save(mean_save_path, self.y_mean)
        np.save(std_save_path, self.y_std)
        np.save(input_mean_save_path, self.x_mean)
        np.save(input_std_save_path, self.x_std)
        
        print(f"\n输入数据已保存到: {X_save_path}")
        print(f"标签数据已保存到: {y_save_path}")
        print(f"日期配对已保存到: {date_pairs_save_path}")
        print(f"输出标准化参数已保存到: {mean_save_path}, {std_save_path}")
        print(f"输入标准化参数已保存到: {input_mean_save_path}, {input_std_save_path}")
        print(f"X数据形状: {normalized_x.shape}")
        print(f"y数据形状: {normalized_y.shape}")
        
        return True
    
    def run(self, window_size=10):
        """运行训练数据生成流程（支持跨周期）"""
        try:
            print("=" * 60)
            print("训练数据生成脚本 (跨周期OHLCV多特征版本)")
            print("=" * 60)
            print(f"  跨周期配置: {self.input_window}个{self.input_frequency} → 预测1个{self.target_frequency}")
            print("=" * 60)
            
            print("\n1. 获取股票池...")
            symbols = self._get_stock_pool()
            if not symbols:
                print("股票池为空，程序退出")
                return False
            
            print("\n2. 下载/加载OHLCV数据...")
            data_dict = self._download_data(symbols)
            if not data_dict:
                print("无数据可用，程序退出")
                return False
            
            if self.input_frequency == self.target_frequency:
                print(f"\n3. 计算OHLCV特征 ({self.input_frequency})...")
                features_dict, dates = self._compute_features(data_dict, self.input_frequency)
                if features_dict is None:
                    print("计算特征失败，程序退出")
                    return False
                
                print("\n4. 构建高维训练数据...")
                market_data, valid_dates = self._build_training_data(features_dict, dates)
                if market_data is None:
                    print("构建训练数据失败，程序退出")
                    return False
                
                print("\n5. 保存训练数据...")
                self._save_training_data(market_data, valid_dates)
                
                print("\n" + "=" * 60)
                print("训练数据生成完成!")
                print("=" * 60)
                print(f"  数据形状: {market_data.shape}")
                print(f"  窗口大小: {window_size}")
                print(f"  输入维度: {window_size * market_data.shape[1]}")
                print(f"  输出维度: {market_data.shape[1]}")
                
                print("\n使用方式:")
                print(f"  import numpy as np")
                print(f"  market_data = np.load('data/training_data.npy')")
                print(f"  dates = np.load('data/training_dates.npy')")
                
                return market_data, valid_dates
            else:
                print(f"\n3. 计算输入特征 ({self.input_frequency})...")
                input_features, input_dates = self._compute_features(data_dict, self.input_frequency)
                if input_features is None:
                    print("计算输入特征失败，程序退出")
                    return False
                
                print(f"\n4. 计算目标特征 ({self.target_frequency})...")
                target_features, target_dates = self._compute_features(data_dict, self.target_frequency)
                if target_features is None:
                    print("计算目标特征失败，程序退出")
                    return False
                
                print("\n5. 构建跨周期训练数据...")
                X_data, y_data, date_pairs = self._build_cross_period_training_data(
                    input_features, input_dates, target_features, target_dates
                )
                if X_data is None:
                    print("构建跨周期训练数据失败，程序退出")
                    return False
                
                print("\n6. 保存训练数据...")
                self._save_cross_period_data(X_data, y_data, date_pairs)
                
                print("\n" + "=" * 60)
                print("跨周期训练数据生成完成!")
                print("=" * 60)
                print(f"  输入频率: {self.input_frequency}, 窗口长度: {self.input_window}")
                print(f"  目标频率: {self.target_frequency}")
                print(f"  输入维度: {X_data.shape[1]}")
                print(f"  输出维度: {y_data.shape[1]}")
                print(f"  样本数量: {len(X_data)}")
                
                print("\n使用方式:")
                print(f"  import numpy as np")
                print(f"  X_data = np.load('data/training_data.npy')")
                print(f"  y_data = np.load('data/training_labels.npy')")
                print(f"  date_pairs = np.load('data/training_date_pairs.npy')")
                
                return X_data, date_pairs
        
        except Exception as e:
            print(f"训练数据生成流程失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description='训练数据生成脚本')
    parser.add_argument('--force-refresh', action='store_true', help='强制刷新缓存，重新下载所有数据')
    parser.add_argument('--window-size', type=int, default=5, help='输入窗口大小')
    args = parser.parse_args()
    
    generator = TrainDataGenerator()
    if args.force_refresh:
        generator.force_refresh = True
        print("⚠️ 强制刷新模式: 将重新下载所有数据")
    
    result = generator.run(window_size=args.window_size)
    
    if isinstance(result, tuple):
        print("\n✅ 训练数据生成成功")
        sys.exit(0)
    else:
        print("\n❌ 训练数据生成失败")
        sys.exit(1)


if __name__ == "__main__":
    main()