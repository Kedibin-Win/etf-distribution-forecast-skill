#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测脚本 - 使用训练好的Transformer模型预测指定日期的市场分布
支持单个标的分析和全市场统计指标评估
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import argparse
import json
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import QMTConfig, DataSourceConfig, TushareConfig

# ---- 统一数据源（方案 A：Tushare 优先 + QMT 可选 + CSV 兜底）----
from data_provider import get_provider

PROVIDER = None  # 延迟初始化


def _get_provider():
    """全局单例数据源（供标的名称/板块列表使用）"""
    global PROVIDER
    if PROVIDER is None:
        PROVIDER = get_provider()
    return PROVIDER


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


FEATURES = ['open_return', 'high_return', 'low_return', 'close_return', 
            'volume_change', 'range_ratio', 'body_ratio', 'upper_shadow', 'lower_shadow',
            'ma5', 'ma10', 'ma20', 'ma60', 'std5', 'std20', 'rsi', 'macd', 'macd_signal',
            'correlation', 'market_rank']
NUM_FEATURES = len(FEATURES)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0)]


class DistributionPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, window_size=60):
        super(DistributionPredictor, self).__init__()
        self.window_size = window_size
        self.d_model = 256
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(512),
            nn.Linear(512, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model, 
                nhead=8, 
                dim_feedforward=1024, 
                dropout=0.3,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ), 
            num_layers=3
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(512),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.output_proj(x)
        return x


def load_model(model_dir='models'):
    model_path = os.path.join(model_dir, 'distribution_predictor.pth')
    config_path = os.path.join(model_dir, 'config.npy')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    config = np.load(config_path, allow_pickle=True).item()
    
    model = DistributionPredictor(
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        window_size=config.get('window_size', 60)
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    
    return model, config


def load_instrument_names(symbols):
    instrument_names = {}

    try:
        provider = _get_provider()
        instrument_names = provider.get_instrument_names(symbols) or {}
    except Exception as e:
        print(f"   通过数据源获取标的名称失败: {e}")

    json_path = 'ETF.json'
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_names = json.load(f)
            for symbol, name in json_names.items():
                if symbol not in instrument_names:
                    instrument_names[symbol] = name
            print(f"   通过JSON文件补充 {len(json_names)} 个标的名称")
        except Exception:
            pass
    
    return instrument_names


def load_training_data(data_dir='data'):
    data_path = os.path.join(data_dir, 'training_data.npy')
    dates_path = os.path.join(data_dir, 'training_dates.npy')
    mean_path = os.path.join(data_dir, 'data_mean.npy')
    std_path = os.path.join(data_dir, 'data_std.npy')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"训练数据不存在: {data_path}")
    
    market_data = np.load(data_path)
    dates = np.load(dates_path) if os.path.exists(dates_path) else None
    
    if dates is not None:
        dates = [d.decode('utf-8') if isinstance(d, bytes) else d for d in dates]
    
    data_mean = np.load(mean_path) if os.path.exists(mean_path) else None
    data_std = np.load(std_path) if os.path.exists(std_path) else None
    
    return market_data, dates, data_mean, data_std


def find_date_index(dates, target_date, target_time=None):
    if dates is None:
        raise ValueError("日期数据未加载")
    
    target_date_str = str(target_date)
    
    if target_time:
        target_full = f"{target_date_str} {target_time}"
        if target_full in dates:
            return dates.index(target_full)
        
        for i, date in enumerate(dates):
            if str(date) == target_full:
                return i
        
        candidates = []
        for i, date in enumerate(dates):
            date_str = str(date)
            if date_str.startswith(target_date_str):
                candidates.append((i, date_str))
        
        if candidates:
            print(f"   精确时点 {target_full} 未找到，查找最接近的时间点")
            target_dt = datetime.strptime(target_full, '%Y-%m-%d %H:%M:%S')
            closest_idx = -1
            min_diff = float('inf')
            for i, dt_str in candidates:
                try:
                    dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                    diff = abs((dt - target_dt).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        closest_idx = i
                except:
                    pass
            
            if closest_idx != -1:
                print(f"   使用最接近的时间点: {candidates[closest_idx][1]} (偏差: {min_diff/60:.1f}分钟)")
                return closest_idx
            else:
                print(f"   找到 {len(candidates)} 个匹配时间点:")
                for i, dt in candidates:
                    print(f"     [{i}] {dt}")
                return candidates[-1][0]
        
        raise ValueError(f"时点 {target_full} 不在训练数据中")
    else:
        if target_date_str in dates:
            return dates.index(target_date_str)
        
        candidates = []
        for i, date in enumerate(dates):
            date_str = str(date)
            if date_str.startswith(target_date_str):
                candidates.append((i, date_str))
        
        if candidates:
            print(f"   找到 {len(candidates)} 个匹配时间点:")
            for i, dt in candidates:
                print(f"     [{i}] {dt}")
            return candidates[-1][0]
        
        last_date_str = str(dates[-1])
        if ' ' in last_date_str:
            last_date_dt = pd.to_datetime(last_date_str.split(' ')[0])
        else:
            last_date_dt = pd.to_datetime(last_date_str)
        target_date_dt = pd.to_datetime(target_date_str)
        
        diff_days = (target_date_dt - last_date_dt).days
        
        if diff_days == 1:
            print(f"   目标日期 {target_date} 不在训练数据中，但可以使用最后 {len(dates)} 天的数据进行预测")
            return len(dates)
        
        raise ValueError(f"日期 {target_date} 不在训练数据中，且不是数据最后一天的下一天")


def extract_feature(data, feature_idx, num_symbols):
    return data[:, feature_idx::num_symbols]


def predict_next_distribution(model, config, historical_data, data_mean=None, data_std=None):
    window_size = config.get('window_size', 60)
    
    if len(historical_data) < window_size:
        raise ValueError(f"历史数据不足，需要至少 {window_size} 天数据，当前 {len(historical_data)} 天")
    
    recent_data = historical_data[-window_size:]
    
    x = torch.tensor(recent_data, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        pred = model(x).cpu().numpy().flatten()
    
    if data_mean is not None and data_std is not None:
        pred = pred * data_std + data_mean
    
    num_symbols = len(pred) // config['num_features']
    
    results = {}
    for feature_name in FEATURES:
        feature_idx = FEATURES.index(feature_name)
        feature_values = extract_feature(pred.reshape(1, -1), feature_idx, len(FEATURES)).flatten()
        results[feature_name] = feature_values
    
    close_return_idx = FEATURES.index('close_return')
    pred_close_rets = results['close_return']
    
    pred_mean = np.mean(pred_close_rets)
    pred_median = np.median(pred_close_rets)
    pred_var_5pct = np.percentile(pred_close_rets, 5)
    pred_var_1pct = np.percentile(pred_close_rets, 1)
    pred_var_95pct = np.percentile(pred_close_rets, 95)
    pred_var_99pct = np.percentile(pred_close_rets, 99)
    pred_std = np.std(pred_close_rets)
    pred_min = np.min(pred_close_rets)
    pred_max = np.max(pred_close_rets)
    
    try:
        pred_skew = np.mean((pred_close_rets - pred_mean) ** 3) / (pred_std ** 3)
        pred_kurtosis = np.mean((pred_close_rets - pred_mean) ** 4) / (pred_std ** 4) - 3
    except:
        pred_skew = 0.0
        pred_kurtosis = 0.0
    
    positive_count = np.sum(pred_close_rets > 0)
    negative_count = np.sum(pred_close_rets < 0)
    positive_ratio = positive_count / num_symbols
    
    sorted_rets = np.sort(pred_close_rets)
    top_n = min(10, num_symbols)
    bottom_n = min(10, num_symbols)
    
    return {
        'predicted_distribution': pred,
        'feature_results': results,
        'close_returns': pred_close_rets,
        'mean': pred_mean,
        'median': pred_median,
        'var_5pct': pred_var_5pct,
        'var_1pct': pred_var_1pct,
        'var_95pct': pred_var_95pct,
        'var_99pct': pred_var_99pct,
        'std': pred_std,
        'min': pred_min,
        'max': pred_max,
        'skew': pred_skew,
        'kurtosis': pred_kurtosis,
        'positive_ratio': positive_ratio,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'num_symbols': num_symbols,
        'top_n_returns': sorted_rets[-top_n:][::-1],
        'bottom_n_returns': sorted_rets[:bottom_n]
    }


def analyze_single_symbol(pred_result, symbol_idx, instrument_names=None, symbols=None):
    results = pred_result['feature_results']
    
    if symbols and symbol_idx < len(symbols):
        code = symbols[symbol_idx]
    else:
        code = f"{symbol_idx + 1:06d}.SH" if (symbol_idx + 1) < 300000 else f"{symbol_idx + 1:06d}.SZ"
    name = instrument_names.get(code, code) if instrument_names else code
    
    close_ret = results['close_return'][symbol_idx]
    open_ret = results['open_return'][symbol_idx]
    high_ret = results['high_return'][symbol_idx]
    low_ret = results['low_return'][symbol_idx]
    volume_change = results['volume_change'][symbol_idx]
    rsi = results['rsi'][symbol_idx]
    macd = results['macd'][symbol_idx]
    ma5 = results['ma5'][symbol_idx]
    ma20 = results['ma20'][symbol_idx]
    
    risk_level = 'LOW'
    if close_ret < -0.02:
        risk_level = 'HIGH'
    elif close_ret < -0.01:
        risk_level = 'MEDIUM'
    
    return {
        'code': code,
        'name': name,
        'close_return': close_ret,
        'open_return': open_ret,
        'high_return': high_ret,
        'low_return': low_ret,
        'range': high_ret - low_ret,
        'volume_change': volume_change,
        'rsi': rsi,
        'macd': macd,
        'ma5': ma5,
        'ma20': ma20,
        'risk_level': risk_level,
        'expected_movement': 'UP' if close_ret > 0 else 'DOWN',
        'strength': 'STRONG' if abs(close_ret) > 0.02 else 'MODERATE' if abs(close_ret) > 0.01 else 'WEAK'
    }


def generate_market_risk_assessment(pred_result):
    mean = pred_result['mean']
    std = pred_result['std']
    var_5pct = pred_result['var_5pct']
    positive_ratio = pred_result['positive_ratio']
    skew = pred_result['skew']
    kurtosis = pred_result['kurtosis']
    
    risk_score = 0
    
    if mean < -0.005:
        risk_score += 30
    elif mean < 0:
        risk_score += 10
    
    if std > 0.02:
        risk_score += 30
    elif std > 0.015:
        risk_score += 15
    
    if var_5pct < -0.05:
        risk_score += 30
    elif var_5pct < -0.03:
        risk_score += 15
    
    if positive_ratio < 0.3:
        risk_score += 20
    elif positive_ratio < 0.45:
        risk_score += 10
    
    if kurtosis > 3:
        risk_score += 15
    
    if skew < -0.5:
        risk_score += 10
    
    risk_score = min(100, risk_score)
    
    if risk_score >= 70:
        risk_level = 'HIGH'
        color = '\033[91m'
    elif risk_score >= 40:
        risk_level = 'MEDIUM'
        color = '\033[93m'
    else:
        risk_level = 'LOW'
        color = '\033[92m'
    
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_color': color,
        'breakdown': {
            'mean_contribution': "下行风险" if mean < -0.005 else "中性" if mean < 0 else "上行机会",
            'volatility_contribution': "高波动" if std > 0.02 else "中等波动" if std > 0.015 else "低波动",
            'tail_risk_contribution': "极端尾部风险" if var_5pct < -0.05 else "中等尾部风险" if var_5pct < -0.03 else "低尾部风险",
            'market_breadth': "市场情绪悲观" if positive_ratio < 0.3 else "市场分化" if positive_ratio < 0.45 else "市场情绪乐观",
            'kurtosis_contribution': "尖峰分布(黑天鹅风险)" if kurtosis > 3 else "正态分布",
            'skew_contribution': "左偏(下行风险大)" if skew < -0.5 else "对称分布"
        }
    }


def generate_trading_signal(pred_result):
    mean = pred_result['mean']
    var_5pct = pred_result['var_5pct']
    std = pred_result['std']
    positive_ratio = pred_result['positive_ratio']
    
    risk_assessment = generate_market_risk_assessment(pred_result)
    
    if risk_assessment['risk_level'] == 'HIGH':
        return 'RISK_OFF', f"风险规避 - 市场风险评分: {risk_assessment['risk_score']}/100, 预测均值: {mean*100:.4f}%"
    
    if mean > 0.002 and positive_ratio > 0.55:
        if var_5pct > -0.03:
            return 'STRONG_BUY', f"强买入信号 - 预测均值: {mean*100:.4f}%, 上涨标的比例: {positive_ratio*100:.1f}%, VaR(5%): {var_5pct*100:.4f}%"
        else:
            return 'CAUTIOUS_BUY', f"谨慎买入 - 预测均值: {mean*100:.4f}%, 但下行风险较大 VaR(5%): {var_5pct*100:.4f}%"
    elif mean > 0.0005:
        return 'WEAK_BUY', f"弱买入信号 - 预测均值: {mean*100:.4f}%, 上涨标的比例: {positive_ratio*100:.1f}%"
    elif mean > -0.001:
        return 'HOLD', f"观望信号 - 预测均值: {mean*100:.4f}%, 市场震荡"
    else:
        return 'SELL', f"卖出信号 - 预测均值: {mean*100:.4f}%, VaR(5%): {var_5pct*100:.4f}%"


def main():
    parser = argparse.ArgumentParser(description='预测指定日期的市场分布')
    parser.add_argument('--date', type=str, required=True, 
                        help='预测日期，格式: YYYY-MM-DD')
    parser.add_argument('--time', type=str, default=None,
                        help='预测时点，格式: HH:MM:SS，例如 14:30:00。如果不指定，默认使用该日期最后一个时间点')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='模型保存目录')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='训练数据目录')
    parser.add_argument('--top-n', type=int, default=10,
                        help='显示表现最好的前N个标的')
    parser.add_argument('--bottom-n', type=int, default=10,
                        help='显示表现最差的前N个标的')
    
    args = parser.parse_args()
    
    target_date = args.date
    target_time = args.time
    
    try:
        datetime.strptime(target_date, '%Y-%m-%d')
    except ValueError:
        print(f"错误: 日期格式不正确，请使用 YYYY-MM-DD 格式")
        return
    
    if target_time:
        try:
            datetime.strptime(target_time, '%H:%M:%S')
        except ValueError:
            print(f"错误: 时间格式不正确，请使用 HH:MM:SS 格式")
            return
    
    if target_time:
        print(f"=" * 70)
        print(f"预测指定时点: {target_date} {target_time}")
        print(f"=" * 70)
    else:
        print(f"=" * 70)
        print(f"预测指定日期: {target_date}")
        print(f"=" * 70)
    
    print("\n1. 加载训练好的模型...")
    model, config = load_model(args.model_dir)
    print(f"   模型配置: input_dim={config['input_dim']}, output_dim={config['output_dim']}, "
          f"window_size={config.get('window_size', 60)}, num_features={config['num_features']}")
    
    print("\n2. 加载训练数据...")
    market_data, dates, data_mean, data_std = load_training_data(args.data_dir)
    print(f"   数据形状: {market_data.shape}")
    if data_mean is not None:
        print(f"   标准化参数已加载")
    print(f"   日期范围: {dates[0]} 到 {dates[-1]}")
    
    print("\n3. 加载标的名称...")
    num_symbols = market_data.shape[1] // NUM_FEATURES
    
    from config import DataConfig
    symbols = DataConfig.stock_pool.get('etf_list', [])
    
    if not symbols:
        symbols = []
        try:
            provider = _get_provider()
            raw_symbols = provider.get_stock_list_in_sector('沪深A股')
            symbols = list(raw_symbols)
            print(f"   通过数据源获取到 {len(symbols)} 个标的代码")
        except Exception as e:
            print(f"   数据源获取股票池失败: {e}")
    else:
        print(f"   从配置文件加载到 {len(symbols)} 个ETF")
    
    if len(symbols) != num_symbols:
        print(f"   警告: 配置股票池数量({len(symbols)})与训练数据数量({num_symbols})不一致")
        symbols = symbols[:num_symbols]
    
    instrument_names = load_instrument_names(symbols)
    print(f"   已加载 {len(instrument_names)} 个标的名称")
    
    print("\n3. 查找目标时点位置...")
    try:
        target_idx = find_date_index(dates, target_date, target_time)
        if target_time:
            print(f"   目标时点 {target_date} {target_time} 在数据中的索引: {target_idx}")
        else:
            print(f"   目标日期 {target_date} 在数据中的索引: {target_idx}")
    except ValueError as e:
        print(f"   错误: {e}")
        print(f"   可用日期范围: {dates[0]} 到 {dates[-1]}")
        return
    
    window_size = config.get('window_size', 60)
    
    if target_idx == len(dates):
        print(f"\n4. 提取输入数据（最后{window_size}个时间点）...")
        start_idx = len(dates) - window_size
        end_idx = len(dates)
        input_data = market_data[start_idx:end_idx]
        print(f"   输入数据范围: {dates[start_idx]} 到 {dates[end_idx-1]}")
        print(f"   输入数据形状: {input_data.shape}")
    else:
        if target_idx < window_size:
            print(f"   错误: 目标时点前只有 {target_idx} 个时间点，需要至少 {window_size} 个")
            return
        
        print(f"\n4. 提取输入数据（前{window_size}个时间点）...")
        start_idx = target_idx - window_size
        end_idx = target_idx
        input_data = market_data[start_idx:end_idx]
        print(f"   输入数据范围: {dates[start_idx]} 到 {dates[end_idx-1]}")
        print(f"   输入数据形状: {input_data.shape}")
    
    print("\n5. 进行预测...")
    result = predict_next_distribution(model, config, input_data, data_mean, data_std)
    
    print("\n" + "=" * 70)
    print("预测结果")
    print("=" * 70)
    if target_time:
        print(f"预测时点: {target_date} {target_time}")
    else:
        print(f"预测日期: {target_date}")
    print(f"标的数量: {result['num_symbols']}")
    
    print("\n" + "-" * 50)
    print("一、全市场统计指标")
    print("-" * 50)
    print(f"预测均值(全市场平均收益): {result['mean']*100:.4f}%")
    print(f"预测中位数: {result['median']*100:.4f}%")
    print(f"预测标准差: {result['std']*100:.4f}%")
    print(f"预测偏度: {result['skew']:.4f} (负值表示左偏，下行风险更大)")
    print(f"预测峰度: {result['kurtosis']:.4f} (正值表示尖峰，极端事件概率更高)")
    
    print("\n" + "-" * 50)
    print("二、风险指标")
    print("-" * 50)
    print(f"VaR(5%): {result['var_5pct']*100:.4f}% (5%概率损失超过此值)")
    print(f"VaR(1%): {result['var_1pct']*100:.4f}% (1%概率损失超过此值)")
    print(f"VaR(95%): {result['var_95pct']*100:.4f}% (5%概率收益超过此值)")
    print(f"VaR(99%): {result['var_99pct']*100:.4f}% (1%概率收益超过此值)")
    print(f"收益区间: [{result['min']*100:.4f}%, {result['max']*100:.4f}%]")
    
    print("\n" + "-" * 50)
    print("三、市场情绪指标")
    print("-" * 50)
    print(f"上涨标的数量: {result['positive_count']}/{result['num_symbols']}")
    print(f"下跌标的数量: {result['negative_count']}/{result['num_symbols']}")
    print(f"上涨比例: {result['positive_ratio']*100:.1f}%")
    
    print("\n" + "-" * 50)
    print("四、市场风险评估")
    print("-" * 50)
    risk_assessment = generate_market_risk_assessment(result)
    print(f"风险评分: {risk_assessment['risk_color']}{risk_assessment['risk_score']}/100\033[0m")
    print(f"风险等级: {risk_assessment['risk_color']}{risk_assessment['risk_level']}\033[0m")
    print("\n风险分项分析:")
    print(f"  • 收益均值: {risk_assessment['breakdown']['mean_contribution']}")
    print(f"  • 波动率: {risk_assessment['breakdown']['volatility_contribution']}")
    print(f"  • 尾部风险: {risk_assessment['breakdown']['tail_risk_contribution']}")
    print(f"  • 市场广度: {risk_assessment['breakdown']['market_breadth']}")
    print(f"  • 分布形态: {risk_assessment['breakdown']['kurtosis_contribution']}")
    print(f"  • 偏度: {risk_assessment['breakdown']['skew_contribution']}")
    
    print("\n" + "-" * 50)
    print(f"五、表现最好的前{args.top_n}个标的")
    print("-" * 50)
    top_indices = np.argsort(result['close_returns'])[-args.top_n:][::-1]
    print(f"{'排名':<4} {'标的名称':<20} {'代码':<12} {'预期收益':<10} {'风险等级':<10} {'强度':<10}")
    print("-" * 75)
    for rank, idx in enumerate(top_indices, 1):
        symbol_analysis = analyze_single_symbol(result, idx, instrument_names, symbols)
        print(f"{rank:<4} {symbol_analysis['name']:<20} {symbol_analysis['code']:<12} "
              f"{symbol_analysis['close_return']*100:>+.4f}%    "
              f"{symbol_analysis['risk_level']:<10} {symbol_analysis['strength']:<10}")
    
    print("\n" + "-" * 50)
    print(f"六、表现最差的前{args.bottom_n}个标的")
    print("-" * 50)
    bottom_indices = np.argsort(result['close_returns'])[:args.bottom_n]
    print(f"{'排名':<4} {'标的名称':<20} {'代码':<12} {'预期收益':<10} {'风险等级':<10} {'强度':<10}")
    print("-" * 75)
    for rank, idx in enumerate(bottom_indices, 1):
        symbol_analysis = analyze_single_symbol(result, idx, instrument_names, symbols)
        print(f"{rank:<4} {symbol_analysis['name']:<20} {symbol_analysis['code']:<12} "
              f"{symbol_analysis['close_return']*100:>+.4f}%    "
              f"{symbol_analysis['risk_level']:<10} {symbol_analysis['strength']:<10}")
    
    print("\n" + "-" * 50)
    print("七、交易信号")
    print("-" * 50)
    signal, reason = generate_trading_signal(result)
    print(f"信号: {signal}")
    print(f"原因: {reason}")
    
    print("\n" + "-" * 50)
    print("八、保存预测结果")
    print("-" * 50)
    
    import pandas as pd
    import os
    
    pred_df = pd.DataFrame()
    pred_df['代码'] = symbols[:result['num_symbols']]
    pred_df['标的名称'] = [instrument_names.get(code, code) for code in symbols[:result['num_symbols']]]
    pred_df['预期收益'] = result['close_returns'] * 100
    
    output_dir = 'predict_output'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'prediction_{target_date}.csv')
    pred_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"   预测结果已保存到: {output_file}")
    
    summary_file = os.path.join(output_dir, f'summary_{target_date}.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"预测日期: {target_date}\n")
        f.write(f"标的数量: {result['num_symbols']}\n")
        f.write(f"预测均值: {result['mean']*100:.4f}%\n")
        f.write(f"预测标准差: {result['std']*100:.4f}%\n")
        f.write(f"VaR(5%): {result['var_5pct']*100:.4f}%\n")
        f.write(f"VaR(1%): {result['var_1pct']*100:.4f}%\n")
        f.write(f"上涨比例: {result['positive_ratio']*100:.1f}%\n")
        f.write(f"交易信号: {signal}\n")
    print(f"   预测摘要已保存到: {summary_file}")
    
    print("\n" + "=" * 70)
    print("预测完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
