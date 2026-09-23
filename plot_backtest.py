# -*- coding: utf-8 -*-
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

for f in ['Microsoft YaHei','SimHei','DengXian']:
    try:
        plt.rcParams['font.sans-serif'] = [f]; break
    except Exception: pass
plt.rcParams['axes.unicode_minus'] = False

d = pd.read_csv('predict_output/black_swan_analysis.csv', encoding='utf-8-sig')
dts = pd.to_datetime(d['日期'])

# ---- 策略：使用真实回测「组合价值」（含成本/调仓限制），口径与 portfolio_stats 一致 ----
strat_curve = d['组合价值'].values / d['组合价值'].values[0] * 100
# ---- 基准：等权 63 ETF 买入持有 ----
bench_curve = np.cumprod(1 + d['真实均值'].values / 100.0) * 100

pos = d['仓位'].values
def mdd(c): return (1 - (c / np.maximum.accumulate(c)).min()) * 100

fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                         gridspec_kw={'height_ratios': [2.2, 1, 1]})
fig.patch.set_facecolor('#0f1117')
for ax in axes:
    ax.set_facecolor('#0f1117')
    ax.tick_params(colors='#c9d1d9', labelsize=9)
    for s in ax.spines.values(): s.set_color('#30363d')
    ax.grid(True, alpha=0.18, color='#8b949e', linestyle='--', linewidth=0.6)

ax = axes[0]
ax.plot(dts, bench_curve, color='#58a6ff', lw=1.8,
        label=f'买入持有 等权63ETF   {bench_curve[-1]-100:+.2f}%   MDD {mdd(bench_curve):.2f}%')
ax.plot(dts, strat_curve, color='#f0883e', lw=2.2,
        label=f'策略(漂移闸门)      {strat_curve[-1]-100:+.2f}%   MDD {mdd(strat_curve):.2f}%')
ax.axhline(100, color='#8b949e', lw=0.8, ls=':')
ax.set_title('策略 vs 基准 · 净值曲线（初始=100，测试集 2024-12-09 ~ 2026-06-24）',
             color='#e6edf3', fontsize=13, pad=10)
ax.set_ylabel('净值', color='#c9d1d9')
ax.legend(loc='upper left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=9.5)

ax = axes[1]
ax.fill_between(dts, 0, pos * 100, color='#3fb950', alpha=0.8, step='mid')
ax.set_ylabel('仓位 %', color='#c9d1d9'); ax.set_ylim(0, 105)
ax.set_title(f'每日仓位（均值 {pos.mean()*100:.1f}%，空仓 {int((pos<0.01).sum())}/{len(d)} 天）',
             color='#e6edf3', fontsize=11, pad=6)

ax = axes[2]
ax.plot(dts, d['分布漂移'], color='#d29922', lw=1.0, label='分布漂移 JS')
ax.plot(dts, d['漂移阈值'], color='#f85149', lw=1.0, ls='--', label='触发阈值')
ax.fill_between(dts, d['分布漂移'], d['漂移阈值'],
                where=(d['分布漂移'] > d['漂移阈值']), color='#f85149', alpha=0.35, interpolate=True)
ax.set_ylabel('JS 散度', color='#c9d1d9')
ax.set_title(f'黑天鹅闸门（触发 {int((d["分布漂移"]>d["漂移阈值"]).sum())}/{len(d)} 天）',
             color='#e6edf3', fontsize=11, pad=6)
ax.legend(loc='upper left', facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=8)

plt.tight_layout()
out = 'predict_output/回测净值对比图.png'
plt.savefig(out, dpi=140, facecolor=fig.get_facecolor(), bbox_inches='tight')
print(f'[OK] {out}')
print(f'策略 {strat_curve[-1]-100:+.2f}% (MDD {mdd(strat_curve):.2f}%) | 基准 {bench_curve[-1]-100:+.2f}% (MDD {mdd(bench_curve):.2f}%)')
