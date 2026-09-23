import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


FEATURES = ['open_return', 'high_return', 'low_return', 'close_return', 
            'volume_change', 'range_ratio', 'body_ratio', 'upper_shadow', 'lower_shadow',
            'ma5', 'ma10', 'ma20', 'ma60', 'std5', 'std20', 'rsi', 'macd', 'macd_signal',
            'correlation', 'market_rank']
NUM_FEATURES = len(FEATURES)


class DistributionDataset(Dataset):
    def __init__(self, data, window_size=10):
        self.data = data
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size - 1
    
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size]
        
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return x_tensor, y_tensor


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


class PositionController:
    def __init__(self, initial_capital=1000000, max_single_risk=0.02, max_total_position=1.0,
                 stop_loss_pct=0.05, take_profit_pct=0.10,
                 open_threshold=0.0005, weak_threshold=0.0001, consecutive_days=2,
                 position_levels=(0, 0.5, 1.0), drift_threshold=0.1,
                 transaction_cost=0.0001, max_position_change=0.3):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_single_risk = max_single_risk
        self.max_total_position = max_total_position
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.open_threshold = open_threshold
        self.weak_threshold = weak_threshold
        self.consecutive_days = consecutive_days
        self.position_levels = position_levels
        self.drift_threshold = drift_threshold
        self.transaction_cost = transaction_cost
        self.max_position_change = max_position_change
        self.current_position_ratio = 0
        self.entry_price = 100
        self.portfolio_values = [initial_capital]
        self.daily_returns = []
        self.position_history = []
        self.signal_history = []
        self.pred_mean_history = []
        self.risk_history = []
        self.total_transaction_costs = 0
        self.trade_count = 0
    
    def calculate_target_position(self, var_5pct, pred_mean, drift_score=0, current_price=100, drift_threshold=0.1):
        if np.isnan(var_5pct) or np.isnan(pred_mean):
            return 0, 0
        
        self.pred_mean_history.append(pred_mean)
        self.risk_history.append(drift_score)
        
        if len(self.pred_mean_history) < self.consecutive_days:
            return 0, 0
        
        recent_means = self.pred_mean_history[-self.consecutive_days:]
        
        if drift_score > drift_threshold:
            position_ratio = self.position_levels[0]
            signal = 0
        elif all(m > self.open_threshold for m in recent_means):
            position_ratio = self.position_levels[2]
            signal = 1
        elif all(m > self.weak_threshold for m in recent_means):
            position_ratio = self.position_levels[1]
            signal = 1
        else:
            position_ratio = self.position_levels[0]
            signal = 0
        
        position_ratio = max(0, min(self.max_total_position, position_ratio))
        
        return position_ratio, signal
    
    def check_stop_loss(self, current_price):
        if self.current_position_ratio == 0:
            return False
        
        price_change = (current_price - self.entry_price) / self.entry_price
        
        if self.current_position_ratio > 0 and price_change < -self.stop_loss_pct:
            return True
        if self.current_position_ratio < 0 and price_change > self.stop_loss_pct:
            return True
        
        return False
    
    def check_take_profit(self, current_price):
        if self.current_position_ratio == 0:
            return False
        
        price_change = (current_price - self.entry_price) / self.entry_price
        
        if self.current_position_ratio > 0 and price_change > self.take_profit_pct:
            return True
        if self.current_position_ratio < 0 and price_change < -self.take_profit_pct:
            return True
        
        return False
    
    def update_position(self, target_position_ratio, current_price):
        self.position_history.append(self.current_position_ratio)
        
        if self.check_stop_loss(current_price):
            self.current_position_ratio = 0
            self.signal_history.append('SL')
            return 'STOP_LOSS'
        
        if self.check_take_profit(current_price):
            self.current_position_ratio = 0
            self.signal_history.append('TP')
            return 'TAKE_PROFIT'
        
        position_diff = target_position_ratio - self.current_position_ratio
        if abs(position_diff) > 0.01:
            max_diff = self.max_position_change
            actual_change = max(-max_diff, min(max_diff, position_diff))
            new_position = self.current_position_ratio + actual_change
            
            if abs(new_position - self.current_position_ratio) > 0.01:
                trade_amount = abs(new_position - self.current_position_ratio) * self.current_capital
                cost = trade_amount * self.transaction_cost
                self.current_capital -= cost
                self.total_transaction_costs += cost
                self.trade_count += 1
                
                self.current_position_ratio = new_position
                self.entry_price = current_price
                self.signal_history.append('TRADE')
                return 'POSITION_CHANGED'
        
        self.signal_history.append('HOLD')
        return 'HOLD'
    
    def update_portfolio(self, daily_return):
        if np.isnan(daily_return) or np.isnan(self.current_position_ratio):
            self.portfolio_values.append(self.current_capital)
            self.daily_returns.append(0)
            return
        self.current_capital = self.current_capital * (1 + daily_return * self.current_position_ratio)
        self.portfolio_values.append(self.current_capital)
        self.daily_returns.append(daily_return)
    
    def get_stats(self):
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        daily_returns_arr = np.array(self.daily_returns)
        position_arr = np.array(self.position_history)
        portfolio_returns = daily_returns_arr * position_arr
        
        sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / (portfolio_returns.std() + 1e-8)
        
        cum_returns = np.cumprod(1 + portfolio_returns)
        max_drawdown = 1 - np.min(cum_returns / np.maximum.accumulate(cum_returns))
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': self.current_capital,
            'total_transaction_costs': self.total_transaction_costs,
            'trade_count': self.trade_count
        }


def load_real_data():
    data_path = os.path.join('data', 'training_data.npy')
    dates_path = os.path.join('data', 'training_dates.npy')
    
    if os.path.exists(data_path):
        print(f"加载真实训练数据: {data_path}")
        market_data = np.load(data_path)
        dates = np.load(dates_path) if os.path.exists(dates_path) else None
        print(f"真实数据形状: {market_data.shape}")
        if dates is not None:
            print(f"日期范围: {dates[0]} 到 {dates[-1]}")
        return market_data, dates
    return None, None


def load_norm_params():
    """
    读取 train_data_generator.py 保存的标准化参数。

    背景：train_data_generator 在保存 training_data.npy 时**已经做过标准化**，
    并把 mean/std 另存为 data/data_mean.npy、data/data_std.npy。
    若 main.py 再调用一次 normalize_data()，即对已标准化数据做「二次标准化」，
    那么 test_data_denorm 还原出的是 z-score 而不是真实收益率，
    回测会把 z-score 当日收益率使用（实测出现 -627% 的"日收益"），结论完全失真。
    """
    mean_path = os.path.join('data', 'data_mean.npy')
    std_path = os.path.join('data', 'data_std.npy')
    if os.path.exists(mean_path) and os.path.exists(std_path):
        return np.load(mean_path), np.load(std_path)
    return None, None


def normalize_data(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True) + 1e-8
    return (data - mean) / std, mean, std


def extract_feature(data, feature_idx, num_symbols):
    """
    抽取某一特征在全部标的上的取值。

    ⚠️ 数据布局为 [sym0_f0..f19, sym1_f0..f19, ..., sym62_f0..f19]，
    因此跨步索引的步长必须是「特征数」NUM_FEATURES，而不是「标的数」。
    调用方应传 NUM_FEATURES(=20)，传 num_symbols(=63) 会把无关列抽出来。
    """
    return data[:, feature_idx::num_symbols]


def kl_divergence(p, q):
    p = p + 1e-10
    q = q + 1e-10
    return np.sum(p * np.log(p / q))


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def compute_distribution_drift(pred_dist, true_dist_history, bins=50):
    hist_pred, _ = np.histogram(pred_dist, bins=bins, density=True)
    hist_pred = hist_pred / (hist_pred.sum() + 1e-10)
    
    drift_scores = []
    for true_dist in true_dist_history:
        hist_true, _ = np.histogram(true_dist, bins=bins, density=True)
        hist_true = hist_true / (hist_true.sum() + 1e-10)
        drift_scores.append(js_divergence(hist_pred, hist_true))
    
    return np.mean(drift_scores), np.max(drift_scores)


def main():
    parser = argparse.ArgumentParser(description='训练分布预测模型')
    parser.add_argument('--no-early-stop', action='store_true', help='取消早停，一直训练到指定轮数')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--gradient-accumulation', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--amp', action='store_true', help='启用自动混合精度训练')
    parser.add_argument('--workers', type=int, default=0, help='DataLoader工作线程数')
    args = parser.parse_args()

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import DataConfig

    np.random.seed(42)
    torch.manual_seed(42)

    WINDOW_SIZE = DataConfig.input_window

    market_data, dates = load_real_data()
    use_real_data = market_data is not None

    if not use_real_data:
        print("未找到真实训练数据，使用模拟数据")
        def simulate_market_data(days=1000, num_symbols=63):
            returns = []
            for day in range(days):
                if (day < 200) or (500 <= day < 700):
                    close_rets = np.random.normal(loc=0.0005, scale=0.01, size=num_symbols)
                else:
                    close_rets = -0.002 + 0.04 * np.random.standard_t(df=3, size=num_symbols)
                
                day_features = []
                for i in range(num_symbols):
                    close_ret = close_rets[i]
                    open_ret = np.random.normal(loc=close_ret * 0.8, scale=0.002)
                    high_ret = max(close_ret, open_ret) + np.random.uniform(0, 0.005)
                    low_ret = min(close_ret, open_ret) - np.random.uniform(0, 0.005)
                    volume_change = np.random.normal(loc=0, scale=0.3)
                    range_ratio = (high_ret - low_ret)
                    body_ratio = np.abs(close_ret - open_ret)
                    upper_shadow = max(0, high_ret - max(open_ret, close_ret))
                    lower_shadow = max(0, min(open_ret, close_ret) - low_ret)
                    day_features.extend([open_ret, high_ret, low_ret, close_ret, 
                                        volume_change, range_ratio, body_ratio, 
                                        upper_shadow, lower_shadow])
                returns.append(day_features)
            return np.array(returns)
        market_data = simulate_market_data()
        market_data, data_mean, data_std = normalize_data(market_data)
    else:
        # 真实数据：复用 train_data_generator 的标准化参数，**避免二次标准化**
        data_mean, data_std = load_norm_params()
        if data_mean is None:
            print("⚠️ 未找到 data/data_mean.npy，回退为就地标准化")
            market_data, data_mean, data_std = normalize_data(market_data)
        else:
            print(f"复用 train_data_generator 标准化参数: mean{data_mean.shape}, std{data_std.shape}")
            print("（已跳过重复标准化，保证反标准化后为真实收益率而非 z-score）")

    print(f"市场数据形状: {market_data.shape}")
    print(f"每日特征维度: {market_data.shape[1]}")

    num_symbols = market_data.shape[1] // NUM_FEATURES
    print(f"ETF数量: {num_symbols}")

    if len(market_data) <= WINDOW_SIZE + 1:
        print(f"错误: 训练数据量不足")
        print(f"  当前数据量: {len(market_data)} 天")
        print(f"  所需数据量: 至少 {WINDOW_SIZE + 2} 天")
        print(f"  请扩大训练数据的时间范围或减小 window_size")
        sys.exit(1)

    total_samples = len(market_data) - WINDOW_SIZE - 1
    train_ratio = 0.7
    val_ratio = 0.15

    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size

    train_data = market_data[:train_size + WINDOW_SIZE + 1]
    val_data = market_data[train_size:train_size + val_size + WINDOW_SIZE + 1]
    test_data = market_data[train_size + val_size:]

    train_dataset = DistributionDataset(train_data, WINDOW_SIZE)
    val_dataset = DistributionDataset(val_data, WINDOW_SIZE)
    test_dataset = DistributionDataset(test_data, WINDOW_SIZE)

    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.workers
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=NUM_WORKERS, pin_memory=pin_memory)

    print(f"数据划分: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("GPU不可用，使用CPU计算")
        print("请确认是否安装了CUDA版本的PyTorch")

    input_dim = market_data.shape[1]
    output_dim = input_dim
    model = DistributionPredictor(input_dim=input_dim, output_dim=output_dim, window_size=WINDOW_SIZE).to(device)

    LR = args.lr
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    warmup_steps = 200
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_steps, eta_min=1e-9)

    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation
    USE_AMP = args.amp
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    print(f"输入维度: {input_dim}")
    print(f"输出维度: {output_dim}")
    print(f"窗口大小: {WINDOW_SIZE}")
    print(f"使用设备: {device}")
    print(f"学习率: {LR}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"有效批次大小: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"自动混合精度: {'启用' if USE_AMP else '禁用'}")
    print(f"DataLoader工作线程: {NUM_WORKERS}")
    print("开始训练分布预测模型...")

    EPOCHS = args.epochs
    PATIENCE = 200
    best_val_loss = float('inf')
    early_stop_counter = 0

    if args.no_early_stop:
        print(f"⚠️ 已取消早停，将训练 {EPOCHS} 轮")

    model_save_dir = 'models'
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, 'distribution_predictor.pth')

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        
        for idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                pred_dist = model(x_batch)
                mse_loss = nn.MSELoss()(pred_dist, y_batch)
                loss = mse_loss
            
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        
        if epoch >= warmup_steps:
            scheduler.step()
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    pred_dist = model(x_batch)
                    mse_loss = nn.MSELoss()(pred_dist, y_batch)
                    loss = mse_loss
                
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), model_save_path)
        elif not args.no_early_stop:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"早停触发，在第 {epoch+1} 轮停止训练")
                break

    config_save_path = os.path.join(model_save_dir, 'config.npy')
    torch.save(model.state_dict(), model_save_path)

    config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'window_size': WINDOW_SIZE,
        'num_features': NUM_FEATURES,
        'data_mean': data_mean,
        'data_std': data_std
    }
    np.save(config_save_path, config)
    print(f"\n模型已保存到: {model_save_path}")
    print(f"配置已保存到: {config_save_path}")


    model.eval()
    model.load_state_dict(torch.load(model_save_path))

    test_predictions = []
    with torch.no_grad():
        for i in range(WINDOW_SIZE + 1, len(test_data)):
            hist = test_data[i-WINDOW_SIZE:i]
            x = torch.tensor(hist, dtype=torch.float32).unsqueeze(0).to(device)
            
            pred = model(x).cpu().numpy().flatten()
            test_predictions.append(pred)

    test_predictions = np.array(test_predictions)

    test_predictions = test_predictions * data_std + data_mean
    test_data_denorm = test_data * data_std + data_mean

    test_dates = None
    if dates is not None:
        test_start_idx = train_size + val_size + WINDOW_SIZE + 1
        test_dates = dates[test_start_idx:test_start_idx + len(test_predictions)]

    print(f"推断ETF数量: {num_symbols}")

    close_return_idx = FEATURES.index('close_return')
    # ⚠️ BUGFIX：第三参数是「跨步长度」，必须传 NUM_FEATURES(20)，不能传 num_symbols(63)。
    # 原代码传 num_symbols 会抽出 20 列互不相关的随机特征，
    # 导致「真实均值 / 预测均值 / 相关性 / 回测绩效」全部失真（曾出现 +20000% 的荒谬收益）。
    true_close_rets = extract_feature(test_data_denorm[WINDOW_SIZE+1:], close_return_idx, NUM_FEATURES)
    pred_close_rets = extract_feature(test_predictions, close_return_idx, NUM_FEATURES)

    true_means = np.mean(true_close_rets, axis=1)
    pred_means = np.mean(pred_close_rets, axis=1)

    pred_left_tail = np.percentile(pred_close_rets, 5, axis=1)
    true_left_tail = np.percentile(true_close_rets, 5, axis=1)

    print(f"\n预测均值统计:")
    print(f"  均值: {np.mean(pred_means):.6f}")
    print(f"  标准差: {np.std(pred_means):.6f}")
    print(f"  最大值: {np.max(pred_means):.6f}")
    print(f"  最小值: {np.min(pred_means):.6f}")

    print(f"\n真实均值统计:")
    print(f"  均值: {np.mean(true_means):.6f}")
    print(f"  标准差: {np.std(true_means):.6f}")
    print(f"  最大值: {np.max(true_means):.6f}")
    print(f"  最小值: {np.min(true_means):.6f}")

    corr_coef = np.corrcoef(pred_means, true_means)[0, 1]
    print(f"\n预测均值与真实均值相关性: {corr_coef:.4f}")

    drift_scores = []
    drift_thresholds = []
    rolling_threshold = 0.1
    for i in range(len(pred_close_rets)):
        start_idx = max(0, i - 30)
        pred_history = [pred_close_rets[j] for j in range(start_idx, i)]
        if len(pred_history) >= 10:
            mean_drift, max_drift = compute_distribution_drift(pred_close_rets[i], pred_history)
        else:
            mean_drift, max_drift = 0, 0
        drift_scores.append(mean_drift)
        
        if i >= 20:
            recent_drifts = drift_scores[-20:]
            rolling_threshold = np.percentile([d for d in recent_drifts if d > 0], 90) if any(d > 0 for d in recent_drifts) else 0.1
        drift_thresholds.append(rolling_threshold)

    drift_scores = np.array(drift_scores)
    drift_thresholds = np.array(drift_thresholds)

    print(f"平均分布漂移阈值: {np.mean(drift_thresholds):.4f}")


    position_controller = PositionController(
        initial_capital=1000000,
        max_single_risk=0.02,
        max_total_position=1.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        open_threshold=0.0005,
        weak_threshold=0.0001,
        drift_threshold=np.mean(drift_thresholds)
    )

    current_price = 100
    for i in range(len(true_means)):
        daily_return = true_means[i]
        
        if i == 0:
            pred_left = 0
            pred_mean = 0
            drift_score = 0
            drift_threshold = 0.1
        else:
            pred_left = pred_left_tail[i-1]
            pred_mean = pred_means[i-1]
            drift_score = drift_scores[i-1] if i-1 < len(drift_scores) else 0
            drift_threshold = drift_thresholds[i-1] if i-1 < len(drift_thresholds) else 0.1
        
        target_position, signal = position_controller.calculate_target_position(
            pred_left, pred_mean, drift_score, current_price, drift_threshold
        )
        
        action = position_controller.update_position(target_position, current_price)
        position_controller.update_portfolio(daily_return)
        
        current_price *= (1 + daily_return)

    portfolio_stats = position_controller.get_stats()
    print("\n=== 策略绩效统计 ===")
    print(f"初始资金: ¥{position_controller.initial_capital:,.2f}")
    print(f"最终资金: ¥{portfolio_stats['final_capital']:,.2f}")
    print(f"总收益率: {portfolio_stats['total_return']*100:.2f}%")
    print(f"夏普比率: {portfolio_stats['sharpe_ratio']:.2f}")
    print(f"最大回撤: {portfolio_stats['max_drawdown']*100:.2f}%")
    print(f"总交易成本: ¥{portfolio_stats['total_transaction_costs']:,.2f}")
    print(f"交易次数: {portfolio_stats['trade_count']}")


    plt.figure(figsize=(18, 24))

    plt.subplot(5, 1, 1)
    if test_dates is not None and len(test_dates) >= len(true_means):
        date_list = [datetime.strptime(str(d)[:10], '%Y-%m-%d') for d in test_dates[:len(true_means)]]
        plt.plot(date_list, true_means, label='True Mean', color='blue', alpha=0.7)
        plt.plot(date_list, pred_means, label='Predicted Mean', color='red', linestyle='--', alpha=0.7)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        plt.gcf().autofmt_xdate()
    else:
        plt.plot(true_means, label='True Mean', color='blue', alpha=0.7)
        plt.plot(pred_means, label='Predicted Mean', color='red', linestyle='--', alpha=0.7)
    plt.title('Market Mean Prediction (Close Returns)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(5, 1, 2)
    if test_dates is not None and len(test_dates) >= len(true_left_tail):
        date_list = [datetime.strptime(str(d)[:10], '%Y-%m-%d') for d in test_dates[:len(true_left_tail)]]
        plt.plot(date_list, true_left_tail, label='True 5% VaR', color='blue', alpha=0.7)
        plt.plot(date_list, pred_left_tail, label='Predicted 5% VaR', color='red', linestyle='--', alpha=0.7)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        plt.gcf().autofmt_xdate()
        plt.fill_between(date_list, pred_left_tail, alpha=0.2, color='red')
    else:
        plt.plot(true_left_tail, label='True 5% VaR', color='blue', alpha=0.7)
        plt.plot(pred_left_tail, label='Predicted 5% VaR', color='red', linestyle='--', alpha=0.7)
        plt.fill_between(range(len(pred_left_tail)), pred_left_tail, alpha=0.2, color='red')
    plt.title('Left-Tail Risk (5% VaR) - Black Swan Early Warning')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(5, 1, 3)
    if test_dates is not None and len(test_dates) >= len(drift_scores):
        date_list = [datetime.strptime(str(d)[:10], '%Y-%m-%d') for d in test_dates[:len(drift_scores)]]
        plt.plot(date_list, drift_scores, label='Distribution Drift (JS)', color='orange', alpha=0.7)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        plt.gcf().autofmt_xdate()
        plt.fill_between(date_list, drift_scores, alpha=0.2, color='orange')
    else:
        plt.plot(drift_scores, label='Distribution Drift (JS)', color='orange', alpha=0.7)
        plt.fill_between(range(len(drift_scores)), drift_scores, alpha=0.2, color='orange')
    plt.axhline(y=drift_threshold, color='red', linestyle='--', alpha=0.7, label='Risk Threshold')
    plt.title('Distribution Drift Detection - Risk Warning Signal')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(5, 1, 4)
    if test_dates is not None and len(test_dates) >= len(position_controller.position_history):
        date_list = [datetime.strptime(str(d)[:10], '%Y-%m-%d') for d in test_dates[:len(position_controller.position_history)]]
        plt.plot(date_list, position_controller.position_history, label='Position', color='purple', alpha=0.7)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        plt.gcf().autofmt_xdate()
        plt.fill_between(date_list, position_controller.position_history, alpha=0.2, color='purple')
    else:
        plt.plot(position_controller.position_history, label='Position', color='purple', alpha=0.7)
        plt.fill_between(range(len(position_controller.position_history)), 
                         position_controller.position_history, alpha=0.2, color='purple')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Full Position')
    plt.title('Position Changes Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(5, 1, 5)
    if test_dates is not None and len(test_dates) >= len(position_controller.portfolio_values):
        date_list = [datetime.strptime(str(d)[:10], '%Y-%m-%d') for d in test_dates[:len(position_controller.portfolio_values)]]
        plt.plot(date_list, position_controller.portfolio_values, label='Portfolio Value', color='green', alpha=0.7)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        plt.gcf().autofmt_xdate()
        plt.fill_between(date_list, position_controller.portfolio_values, 
                         position_controller.initial_capital, alpha=0.2, color='green')
    else:
        plt.plot(position_controller.portfolio_values, label='Portfolio Value', color='green', alpha=0.7)
        plt.fill_between(range(len(position_controller.portfolio_values)), 
                         position_controller.portfolio_values, 
                         position_controller.initial_capital, 
                         alpha=0.2, color='green')
    plt.axhline(y=position_controller.initial_capital, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
    plt.title(f'Portfolio Performance (Test Set) - Total Return: {portfolio_stats["total_return"]*100:.2f}%, Max DD: {portfolio_stats["max_drawdown"]*100:.2f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n=== 保存黑天鹅分析数据 ===")
    
    output_dir = 'predict_output'
    os.makedirs(output_dir, exist_ok=True)
    
    black_swan_df = pd.DataFrame()
    if test_dates is not None:
        black_swan_df['日期'] = [str(d)[:10] for d in test_dates[:len(true_means)]]
    else:
        black_swan_df['日期'] = range(len(true_means))
    
    black_swan_df['真实均值'] = true_means * 100
    black_swan_df['预测均值'] = pred_means * 100
    black_swan_df['真实VaR5%'] = true_left_tail * 100
    black_swan_df['预测VaR5%'] = pred_left_tail * 100
    black_swan_df['分布漂移'] = drift_scores
    black_swan_df['漂移阈值'] = drift_thresholds
    black_swan_df['仓位'] = position_controller.position_history[:len(true_means)]
    # 修复：portfolio_values[0] 是初始资金，其后每日追加一个值（共 len(true_means)+1 个）。
    # 原写法 [:len(true_means)] 会丢掉最后一天，导致整列相对日期滞后一日。
    _pv = position_controller.portfolio_values
    black_swan_df['组合价值'] = _pv[1:len(true_means) + 1] if len(_pv) == len(true_means) + 1 else _pv[:len(true_means)]
    
    black_swan_file = os.path.join(output_dir, 'black_swan_analysis.csv')
    black_swan_df.to_csv(black_swan_file, index=False, encoding='utf-8-sig')
    print(f"黑天鹅分析数据已保存到: {black_swan_file}")
    
    portfolio_stats_file = os.path.join(output_dir, 'portfolio_stats.txt')
    with open(portfolio_stats_file, 'w', encoding='utf-8') as f:
        f.write(f"初始资金: ¥{position_controller.initial_capital:,.2f}\n")
        f.write(f"最终资金: ¥{portfolio_stats['final_capital']:,.2f}\n")
        f.write(f"总收益率: {portfolio_stats['total_return']*100:.2f}%\n")
        f.write(f"夏普比率: {portfolio_stats['sharpe_ratio']:.2f}\n")
        f.write(f"最大回撤: {portfolio_stats['max_drawdown']*100:.2f}%\n")
        f.write(f"总交易成本: ¥{portfolio_stats['total_transaction_costs']:,.2f}\n")
        f.write(f"交易次数: {portfolio_stats['trade_count']}\n")
    print(f"组合统计数据已保存到: {portfolio_stats_file}")


if __name__ == '__main__':
    main()