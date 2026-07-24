import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ---------- 缓存量子模拟 ----------
@st.cache_data(show_spinner=False)
def run_grover_simulation(n, target_state, iterations, shots=512):
    qc = QuantumCircuit(n, n)
    for q in range(n):
        qc.h(q)
    target_bin = format(target_state, f'0{n}b')[::-1]
    for _ in range(iterations):
        for q, bit in enumerate(target_bin):
            if bit == '0':
                qc.x(q)
        if n == 2:
            qc.cz(0, 1)
        elif n == 3:
            qc.h(2)
            qc.ccx(0, 1, 2)
            qc.h(2)
        for q, bit in enumerate(target_bin):
            if bit == '0':
                qc.x(q)
        for q in range(n):
            qc.h(q)
            qc.x(q)
        if n == 2:
            qc.cz(0, 1)
        elif n == 3:
            qc.h(2)
            qc.ccx(0, 1, 2)
            qc.h(2)
        for q in range(n):
            qc.x(q)
            qc.h(q)
    qc.measure(range(n), range(n))
    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts

# ---------- Plotly 版几何圆盘 ----------
def plot_bloch_disk(n, iterations):
    N = 2**n
    theta = np.arcsin(1/np.sqrt(N))
    t_angle = np.pi/2
    s_angle = np.pi/2 - theta
    current_angle = np.pi/2 - (theta + 2*theta*iterations)

    # 创建图形
    fig = go.Figure()
    # 单位圆
    circle_x = np.cos(np.linspace(0, 2*np.pi, 200))
    circle_y = np.sin(np.linspace(0, 2*np.pi, 200))
    fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines',
                             line=dict(color='gray', width=1), showlegend=False))
    # 坐标轴
    fig.add_trace(go.Scatter(x=[-1.2,1.2], y=[0,0], mode='lines',
                             line=dict(color='black', width=0.5), showlegend=False))
    fig.add_trace(go.Scatter(x=[0,0], y=[-1.2,1.2], mode='lines',
                             line=dict(color='black', width=0.5), showlegend=False))
    # |t⟩ 箭头
    fig.add_annotation(x=0, y=1, ax=0, ay=0, xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='blue',
                       text='|t⟩', font=dict(color='blue', size=14))
    # |s⟩ 箭头
    fig.add_annotation(x=np.cos(s_angle), y=np.sin(s_angle), ax=0, ay=0,
                       xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='green',
                       text='|s⟩', font=dict(color='green', size=12))
    # 当前态箭头
    fig.add_annotation(x=np.cos(current_angle), y=np.sin(current_angle), ax=0, ay=0,
                       xref='x', yref='y', axref='x', ayref='y',
                       showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=3, arrowcolor='red',
                       text='当前态', font=dict(color='red', size=12))

    fig.update_layout(width=450, height=450, title=f'几何圆盘 (迭代{iterations}次)',
                      xaxis=dict(range=[-1.3,1.3], zeroline=False, showgrid=False),
                      yaxis=dict(range=[-1.3,1.3], zeroline=False, showgrid=False),
                      showlegend=False)
    return fig

# ---------- 界面 ----------
st.title('Grover 量子搜索算法可视化（Plotly 稳定版）')
st.markdown('包含：测量分布、几何圆盘、迭代成功率、概率幅、复杂度对比。')

n = st.sidebar.selectbox('量子比特数 (搜索空间大小)', [2, 3], index=0)
N = 2**n
target = st.sidebar.selectbox('选择目标态',
                              list(range(N)),
                              format_func=lambda x: f'|{x:0{n}b}⟩')
run = st.sidebar.button('▶️ 运行完整分析')

if run:
    optimal_k = int(np.floor(np.pi/4 * np.sqrt(N)))
    target_str = f'{target:0{n}b}'
    shots = 512

    # 1. 最优迭代结果
    counts_opt = run_grover_simulation(n, target, optimal_k, shots)
    success_opt = counts_opt.get(target_str, 0) / shots

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('测量结果分布')
        df_counts = pd.DataFrame({'次数': counts_opt})
        st.bar_chart(df_counts, use_container_width=True)
    with col2:
        st.subheader('振幅放大几何图像')
        c1, c2 = st.columns(2)
        with c1:
            st.caption('初始 (k=0)')
            st.plotly_chart(plot_bloch_disk(n, 0), use_container_width=True)
        with c2:
            st.caption(f'最优 (k={optimal_k})')
            st.plotly_chart(plot_bloch_disk(n, optimal_k), use_container_width=True)

    st.metric(label=f'目标态 |{target_str}⟩ 成功率', value=f'{success_opt:.1%}')

    # 2. 迭代成功率
    st.markdown('---')
    st.header('📈 迭代次数与成功率')
    rates = {}
    for k in range(4):
        counts_k = run_grover_simulation(n, target, k, shots)
        rates[k] = counts_k.get(target_str, 0) / shots
    df_rate = pd.DataFrame({'成功率': rates})
    st.line_chart(df_rate)

    # 3. 概率幅条形图（用测量结果计算模长）
    st.markdown('---')
    st.header('🔬 概率幅模长（基于测量结果）')
    total = sum(counts_opt.values())
    states = [f'|{i:0{n}b}⟩' for i in range(N)]
    amplitudes = [np.sqrt(counts_opt.get(s, 0) / total) for s in states]
    df_amp = pd.DataFrame({'概率幅模长': amplitudes}, index=states)
    st.bar_chart(df_amp)

    # 4. 复杂度对比
    st.markdown('---')
    st.header('⚖️ 经典 vs 量子查询复杂度')
    classical_queries = N - 1
    quantum_queries = optimal_k
    st.write(f'经典最坏查询次数：**{classical_queries}**，Grover 最优迭代次数：**{quantum_queries}**')
    df_cmp = pd.DataFrame({'查询次数': [classical_queries, quantum_queries]},
                          index=['经典搜索', 'Grover算法'])
    st.bar_chart(df_cmp, use_container_width=True)

    st.info(f'量子加速：从 {classical_queries} 次查询降至 {quantum_queries} 次迭代。')

else:
    st.info('请在左侧选择参数后点击运行按钮。')
