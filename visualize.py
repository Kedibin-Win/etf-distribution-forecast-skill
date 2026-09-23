import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

COLOR_SCALE = [
    [0.0, "#00ff00"],
    [0.3, "#88ff88"],
    [0.45, "#dfffdf"],
    [0.5, "#ffffff"],
    [0.55, "#ffe5e5"],
    [0.7, "#ff8888"],
    [1.0, "#ff0000"]
]


def get_etf_sector(code):
    sector_mapping = {
        '512010': '医药医疗', '159929': '医药医疗', '512290': '医药医疗', '159647': '医药医疗',
        '512660': '军工', '512670': '军工', '161024': '军工',
        '512000': '券商', '512880': '券商', '510230': '金融', '512800': '银行',
        '512980': '传媒', '516160': '新能源', '159864': '新能源',
        '159995': '芯片', '512760': '芯片', '512480': '半导体',
        '516880': '科创', '588000': '科创50',
        '159915': '创业板', '159998': '计算机', '159852': '软件',
        '512170': '电子', '159939': '信息技术',
        '510300': '沪深300', '510500': '中证500', '512500': '中证500',
        '159919': '沪深300', '510880': '红利',
        '512580': '有色', '159934': '有色', '516010': '能源',
        '159996': '家电', '159928': '食品', '512690': '酒',
        '510330': 'ETF', '511260': '债券',
        '159840': '锂电池', '159865': '养殖', '159611': '电力',
        '562570': '信创', '562500': '机器人', '159638': '高端装备',
    }
    base_code = code[:6]
    return sector_mapping.get(base_code, '其他')


def generate_heatmap(df):
    fig = px.treemap(
        df,
        path=['行业', '标的名称'],
        values='强度',
        color='预期收益',
        color_continuous_scale=COLOR_SCALE,
        range_color=[-max(abs(df['预期收益'].min()), abs(df['预期收益'].max())),
                 max(abs(df['预期收益'].min()), abs(df['预期收益'].max()))],
        color_continuous_midpoint=0,
        branchvalues='total',
        hover_data={
            '代码': ':',
            '预期收益': ':%.2f',
            '强度': ':%.2f',
            '风险等级': ':'
        },
        height=900
    )
    
    fig.update_traces(
        texttemplate=(
            "<b>%{label}</b><br>"
            "📈%{customdata[1]:+.2f}%"
        ),
        hovertemplate=( 
            "<b>%{label}</b><br>"
            "代码: %{customdata[0]}<br>"
            "预期收益: <b>%{customdata[1]:+.2f}%</b><br>"
            "强度: %{customdata[2]:.2f}<br>"
            "风险等级: %{customdata[3]}"
        ),
        textfont=dict(size=16, color='black')
    )
    
    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        coloraxis_colorbar=dict(
            title="预期收益(%)",
            ticks="inside",
            thickness=20,
            len=0.6,
            y=0.7
        )
    )
    return fig


def display_black_swan_analysis():
    st.subheader("🦢 黑天鹅分析")
    
    black_swan_file = os.path.join('predict_output', 'black_swan_analysis.csv')
    
    if not os.path.exists(black_swan_file):
        st.warning("⚠️ 黑天鹅分析数据不存在，请先运行 `uv run python main.py` 训练模型")
        return
    
    try:
        bs_df = pd.read_csv(black_swan_file, encoding='utf-8-sig')
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=bs_df['日期'], y=bs_df['真实均值'], name='真实均值', line=dict(color='blue', width=2)))
        fig1.add_trace(go.Scatter(x=bs_df['日期'], y=bs_df['预测均值'], name='预测均值', line=dict(color='red', width=2, dash='dash')))
        fig1.update_layout(title='市场均值预测', xaxis_title='日期', yaxis_title='收益率(%)', height=400)
        st.plotly_chart(fig1, width='stretch')
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=bs_df['日期'], y=bs_df['真实VaR5%'], name='真实VaR5%', line=dict(color='blue', width=2)))
        fig2.add_trace(go.Scatter(x=bs_df['日期'], y=bs_df['预测VaR5%'], name='预测VaR5%', line=dict(color='red', width=2, dash='dash')))
        fig2.add_hline(y=0, line_dash="dot", line_color="gray")
        fig2.update_layout(title='左尾风险 (5% VaR) - 黑天鹅预警', xaxis_title='日期', yaxis_title='VaR5%(%)', height=400)
        st.plotly_chart(fig2, width='stretch')
        
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=bs_df['日期'], y=bs_df['分布漂移'], name='分布漂移(JS)', line=dict(color='orange', width=2)))
        fig3.add_trace(go.Scatter(x=bs_df['日期'], y=bs_df['漂移阈值'], name='风险阈值', line=dict(color='red', width=2, dash='dash')))
        fig3.update_layout(title='分布漂移检测 - 风险预警信号', xaxis_title='日期', yaxis_title='漂移分数', height=400)
        st.plotly_chart(fig3, width='stretch')
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=bs_df['日期'], y=bs_df['仓位'], name='仓位', line=dict(color='purple', width=2)))
        fig4.add_hline(y=0, line_dash="dot", line_color="gray")
        fig4.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="满仓")
        fig4.update_layout(title='仓位变化', xaxis_title='日期', yaxis_title='仓位', height=400)
        st.plotly_chart(fig4, width='stretch')
        
    except Exception as e:
        st.error(f"❌ 加载黑天鹅分析数据失败: {str(e)}")


def main():
    st.set_page_config(layout="wide")
    
    st.title("📊 大盘预测热力图")
    
    output_dir = 'predict_output'
    
    if not os.path.exists(output_dir):
        st.error(f"❌ 预测输出目录不存在: {output_dir}")
        st.info("请先运行 `uv run python predict.py --date 2026-06-24` 生成预测数据")
        return
    
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and 'black_swan' not in f]
    if not csv_files:
        st.error(f"❌ {output_dir} 目录中没有预测CSV文件")
        st.info("请先运行 `uv run python predict.py --date 2026-06-24` 生成预测数据")
        return
    
    selected_file = st.selectbox("选择预测日期", csv_files)
    csv_path = os.path.join(output_dir, selected_file)
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        df['行业'] = df['代码'].apply(get_etf_sector)
        df['强度'] = df['预期收益'].abs()
        df['风险等级'] = df['预期收益'].apply(
            lambda x: 'HIGH' if x < -2 else ('MEDIUM' if x < -1 else 'LOW')
        )
        
        st.plotly_chart(generate_heatmap(df), width='stretch')
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🔥 最强收益", 
                   f"{df['预期收益'].max():+.2f}%",
                   df.loc[df['预期收益'].idxmax(), '标的名称'])
        col2.metric("💧 最大亏损", 
                   f"{df['预期收益'].min():+.2f}%",
                   df.loc[df['预期收益'].idxmin(), '标的名称'])
        col3.metric("⚖️ 多空比", 
                   f"{len(df[df['预期收益']>0])}:{len(df[df['预期收益']<0])}",
                   f"均值 {df['预期收益'].mean():+.2f}%")
        
        st.subheader("📋 详细预测列表")
        st.dataframe(df.sort_values('预期收益', ascending=False), width='stretch')
        
    except Exception as e:
        st.error(f"❌ 加载数据失败: {str(e)}")
    
    display_black_swan_analysis()


if __name__ == "__main__":
    main()