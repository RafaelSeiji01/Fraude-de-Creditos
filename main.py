import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# -------------------------------------------------------------
# 1. CONFIGURAÇÃO DA PÁGINA E DESIGN SYSTEM
# -------------------------------------------------------------
st.set_page_config(
    page_title="Monitoramento Antifraude em Cartões",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp {
            background-color: #f8f9fa;
        }
        .metric-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 16px 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border: 1px solid #e9ecef;
            margin-bottom: 15px;
        }
        .metric-card h6 {
            color: #6c757d;
            font-size: 0.85rem;
            margin-bottom: 5px;
            font-weight: 600;
        }
        .metric-card h3 {
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 0;
        }
        .question-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 24px 28px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border: 1px solid #e9ecef;
            margin-bottom: 24px;
        }
        .question-title {
            font-size: 1.18rem;
            font-weight: 700;
            color: #212529;
            margin-bottom: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. CARGA E TRATAMENTO DOS DADOS (creditcard.csv)
# -------------------------------------------------------------
@st.cache_data
def carregar_dados():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    
    possiveis_caminhos = [
        os.path.join(diretorio_atual, 'creditcard.parquet'),
        os.path.join(diretorio_atual, 'archive', 'creditcard.parquet'),
        os.path.join(os.path.dirname(diretorio_atual), 'creditcard.parquet'),
        # Fallbacks caso mantenha o zip
        os.path.join(diretorio_atual, 'creditcard.zip'),
        os.path.join(diretorio_atual, 'archive', 'creditcard.zip'),
    ]
    
    caminho_arquivo = next((p for p in possiveis_caminhos if os.path.exists(p)), None)
    
    if caminho_arquivo is None:
        st.error("Arquivo de dados não foi encontrado!")
        st.stop()
        
    if caminho_arquivo.endswith('.parquet'):
        return pd.read_parquet(caminho_arquivo)
    
    # Fallback para ZIP/CSV caso não tenha convertido ainda
    df = pd.read_csv(caminho_arquivo, compression='zip' if caminho_arquivo.endswith('.zip') else None)
    
    # Redução de tipos em runtime para não esgotar a RAM
    float_cols = [c for c in df.columns if df[c].dtype == 'float64']
    df[float_cols] = df[float_cols].astype('float32')
    df['Class'] = df['Class'].astype('int8')
    
    df['Status'] = df['Class'].map({0: 'Legítima', 1: 'Fraude'}).astype('category')
    df['Hora_do_Dia'] = ((df['Time'] / 3600) % 24).astype('int8')

    bins_valor = [-1, 10, 50, 200, 500, 1000, float('inf')]
    labels_valor = ['$0 - $10', '$10 - $50', '$50 - $200', '$200 - $500', '$500 - $1.000', '> $1.000']
    df['Faixa_Valor'] = pd.cut(df['Amount'], bins=bins_valor, labels=labels_valor)
    
    return df

df = carregar_dados()

# Estatísticas Gerais
total_transacoes = len(df)
total_fraudes = int((df['Class'] == 1).sum())
total_legitimas = int((df['Class'] == 0).sum())
taxa_fraude = (total_fraudes / total_transacoes) * 100
prejuizo_total = df[df['Class'] == 1]['Amount'].sum()

# Cálculo do Intervalo de Confiança de 95% para Médias (Z = 1.96)
valores_leg = df[df['Class'] == 0]['Amount']
valores_frd = df[df['Class'] == 1]['Amount']

media_legitima = valores_leg.mean()
n_leg = len(valores_leg)
erro_leg = 1.96 * (valores_leg.std() / np.sqrt(n_leg))

media_fraude = valores_frd.mean()
n_frd = len(valores_frd)
erro_frd = 1.96 * (valores_frd.std() / np.sqrt(n_frd))

# Amostra balanceada para o gráfico PCA
amostra_legitima = df[df['Class'] == 0].sample(n=min(3000, total_legitimas), random_state=42)
amostra_fraude = df[df['Class'] == 1]
df_amostra = pd.concat([amostra_legitima, amostra_fraude])

CORES = {'Legítima': '#0d6efd', 'Fraude': '#dc3545'}

# -------------------------------------------------------------
# 3. CABEÇALHO E KPIS DE TOPO
# -------------------------------------------------------------
st.title("Diagnóstico e Monitoramento de Fraudes em Cartão de Crédito")
st.caption("Painel analítico investigativo com inferência estatística e modelagem preditiva")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
        <div class="metric-card">
            <h6>TOTAL DE TRANSAÇÕES</h6>
            <h3 style="color: #0d6efd;">{total_transacoes:,}</h3>
        </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
        <div class="metric-card">
            <h6>PREJUÍZO ACUMULADO</h6>
            <h3 style="color: #dc3545;">$ {prejuizo_total:,.2f}</h3>
        </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
        <div class="metric-card">
            <h6>GOLPES CONFIRMADOS</h6>
            <h3 style="color: #dc3545;">{total_fraudes:,}</h3>
        </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown(f"""
        <div class="metric-card">
            <h6>TAXA DE INCIDÊNCIA</h6>
            <h3 style="color: #ffc107;">{taxa_fraude:.3f}%</h3>
        </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# 4. OS 5 BLOCOS DE DIAGNÓSTICO ESTATÍSTICO
# -------------------------------------------------------------

# BLOCO 1: Comportamento Financeiro e Intervalo de Confiança da Média
st.markdown('<div class="question-card">', unsafe_allow_html=True)
col_texto_1, col_graf_1 = st.columns([5, 7])
with col_texto_1:
    st.markdown('<div class="question-title">1. Qual o comportamento financeiro e o valor das fraudes?</div>', unsafe_allow_html=True)
    st.write("Estimativa da média populacional com **Intervalo de Confiança de 95%**:")
    
    st.latex(r"\text{IC}_{\text{Média}} = \bar{X} \pm 1.96 \cdot \left(\frac{s}{\sqrt{n}}\right)")
    
    st.markdown(f"""
    * **Legítima (n = {n_leg:,}):** Média **${media_legitima:.2f}** (±${erro_leg:.2f})
    * **Fraude (n = {n_frd:,}):** Média **${media_fraude:.2f}** (±${erro_frd:.2f})
    """)
    
    st.info(f"""
    **Por que a margem da fraude é muito maior?**
    * **Legítima (Margem minúscula de ±${erro_leg:.2f}):** A amostra é massiva (n = {n_leg:,}). O tamanho amostral no denominador divide a incerteza quase a zero.
    * **Fraude (Margem ampla de ±${erro_frd:.2f}):** A amostra é pequena (n = {n_frd:,}) e a variação de valores é extrema (desde centavos até milhares de dólares). Menos dados com alta dispersão geram maior incerteza amostral.
    """)

with col_graf_1:
    fig_ic_media = go.Figure()
    fig_ic_media.add_trace(go.Bar(
        name='Legítima',
        x=['Legítima'],
        y=[media_legitima],
        error_y=dict(type='data', array=[erro_leg], visible=True, thickness=2, width=6),
        marker_color=CORES['Legítima'],
        text=[f"${media_legitima:.2f}<br>±${erro_leg:.2f}"],
        textposition='outside'
    ))
    fig_ic_media.add_trace(go.Bar(
        name='Fraude',
        x=['Fraude'],
        y=[media_fraude],
        error_y=dict(type='data', array=[erro_frd], visible=True, thickness=2, width=6),
        marker_color=CORES['Fraude'],
        text=[f"${media_fraude:.2f}<br>±${erro_frd:.2f}"],
        textposition='outside'
    ))
    fig_ic_media.update_layout(
        template='plotly_white',
        title='Ticket Médio com Margem de Erro (IC 95%)',
        height=340,
        yaxis=dict(title='Média ($)', range=[0, (media_fraude + erro_frd) * 1.35]),
        xaxis=dict(title=''),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=30)
    )
    st.plotly_chart(fig_ic_media, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# BLOCO 2: Risco Relativo por Faixa de Valor e IC Proporção
st.markdown('<div class="question-card">', unsafe_allow_html=True)
col_texto_2, col_graf_2 = st.columns([5, 7])
with col_texto_2:
    st.markdown('<div class="question-title">2. Em quais faixas de valor o risco de fraude é proporcionalmente maior?</div>', unsafe_allow_html=True)
    st.write("Taxa de fraude por categoria de valor com **margem de erro proporcional de 95%**:")
    
    st.latex(r"\text{IC}_{\text{Proporção}} = \hat{p} \pm 1.96 \cdot \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}")
    
    st.info("""
    **Por que o tamanho das barras de erro varia por faixa?**
    * **Faixas baixas ($0 a $50):** Margem de erro estreita devido à concentração de centenas de milhares de transações.
    * **Faixas altas (> $1.000):** Barra de erro bem mais longa porque pouquíssimas transações atingem esse valor, aumentando a margem de incerteza da proporção estimada.
    """)

with col_graf_2:
    df_faixas = df.groupby('Faixa_Valor', observed=False).agg(
        Total=('Class', 'count'),
        Fraudes=('Class', 'sum')
    ).reset_index()

    p = np.where(df_faixas['Total'] > 0, df_faixas['Fraudes'] / df_faixas['Total'], 0.0)
    n = df_faixas['Total'].values
    se = np.where(n > 0, np.sqrt((p * (1 - p)) / n), 0.0)
    margem_erro_pct = 1.96 * se * 100

    df_faixas['Taxa_Fraude_Pct'] = p * 100
    df_faixas['Margem_Erro_Pct'] = margem_erro_pct

    fig_faixas = go.Figure()
    fig_faixas.add_trace(go.Bar(
        x=df_faixas['Faixa_Valor'],
        y=df_faixas['Taxa_Fraude_Pct'],
        error_y=dict(type='data', array=df_faixas['Margem_Erro_Pct'], visible=True, thickness=1.5, width=6),
        marker_color='#dc3545',
        text=[f"{val:.2f}%<br>±{err:.2f}%" for val, err in zip(df_faixas['Taxa_Fraude_Pct'], df_faixas['Margem_Erro_Pct'])],
        textposition='outside'
    ))
    max_y = (df_faixas['Taxa_Fraude_Pct'] + df_faixas['Margem_Erro_Pct']).max()
    fig_faixas.update_layout(
        template='plotly_white',
        title='Taxa de Fraude com Margem de Erro (IC 95%)',
        height=340,
        xaxis=dict(title='Faixa de Valor'),
        yaxis=dict(title='Taxa de Fraude (%)', range=[0, max(max_y * 1.35, 1.0)]),
        margin=dict(l=40, r=40, t=20, b=30)
    )
    st.plotly_chart(fig_faixas, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# BLOCO 3: Horários em Ciclo Diário de 24 Horas
st.markdown('<div class="question-card">', unsafe_allow_html=True)
col_texto_3, col_graf_3 = st.columns([5, 7])
with col_texto_3:
    st.markdown('<div class="question-title">3. Em que horários as fraudes mais acontecem no ciclo diário?</div>', unsafe_allow_html=True)
    st.write("Densidade comparativa de transações consolidadas em 24 horas (00h às 23h).")
    st.markdown("""
    * **Transações legítimas:** Caem expressivamente na madrugada (entre 01h e 06h).
    * **Fraudes:** Mantêm taxa constante na madrugada, aproveitando o período de menor monitoramento dos correntistas.
    """)

with col_graf_3:
    fig_tempo = px.histogram(
        df,
        x='Hora_do_Dia',
        color='Status',
        barmode='overlay',
        nbins=24,
        color_discrete_map=CORES,
        histnorm='probability density'
    )
    fig_tempo.update_layout(
        template='plotly_white',
        height=320,
        xaxis=dict(title='Hora do Dia (00:00 às 23:00)', tickmode='linear', tick0=0, dtick=2, range=[-0.5, 23.5]),
        yaxis=dict(title='Densidade'),
        legend=dict(title='', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=20, b=30)
    )
    st.plotly_chart(fig_tempo, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# BLOCO 4: Espaço Multidimensional PCA
st.markdown('<div class="question-card">', unsafe_allow_html=True)
col_texto_4, col_graf_4 = st.columns([5, 7])
with col_texto_4:
    st.markdown('<div class="question-title">4. As variáveis PCA conseguem isolar transações suspeitas?</div>', unsafe_allow_html=True)
    st.write("Selecione os eixos cartesianos para avaliar a separabilidade espacial entre compras legítimas e golpes:")
    
    c_pca1, c_pca2 = st.columns(2)
    with c_pca1:
        col_x = st.selectbox("Eixo X (Horizontal):", [f'V{i}' for i in range(1, 29)], index=13)
    with c_pca2:
        col_y = st.selectbox("Eixo Y (Vertical):", [f'V{i}' for i in range(1, 29)], index=16)

    st.info("""
    **O que representam esses eixos (V1 a V28)?**
    * **Anonimização via PCA:** Para proteger o sigilo bancário (dados de cartão, geolocalização e histórico), a técnica de Análise de Componentes Principais compactou dezenas de variáveis reais nesses eixos numéricos.
    * **Coordenadas de Risco:** Cada ponto no gráfico é uma transação posicionada conforme seu padrão comportamental.
    * **Poder Preditivo:** Cruzamentos como **V14 × V17** isolam as fraudes (vermelho) nas extremidades, indicando aos modelos de Machine Learning os principais padrões de corte.
    """)

with col_graf_4:
    fig_pca = px.scatter(
        df_amostra,
        x=col_x,
        y=col_y,
        color='Status',
        color_discrete_map=CORES,
        opacity=0.65,
        hover_data=['Amount', 'Hora_do_Dia']
    )
    fig_pca.update_layout(
        template='plotly_white',
        title=f'Projeção Espacial 2D: {col_x} vs {col_y}',
        height=340,
        legend=dict(title='', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=40, b=30)
    )
    st.plotly_chart(fig_pca, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# BLOCO 5: Desbalanceamento Severo de Classes
st.markdown('<div class="question-card">', unsafe_allow_html=True)
col_texto_5, col_graf_5 = st.columns([5, 7])
with col_texto_5:
    st.markdown('<div class="question-title">5. Como o desbalanceamento severo impacta a modelagem?</div>', unsafe_allow_html=True)
    st.write("Avaliação da representatividade da classe minoritária:")
    st.markdown(f"""
    * Apenas **{taxa_fraude:.3f}%** da base é composta por golpes (1 fraude para cada ~578 compras legítimas).
    * A acurácia tradicional é inadequada para medir o desempenho do classificador.
    * A métrica principal de avaliação deve ser **PR-AUC** (Precision-Recall AUC) e **F1-Score**.
    """)

with col_graf_5:
    df_prop = df['Status'].value_counts().reset_index()
    df_prop.columns = ['Status', 'Total']
    fig_pie = px.pie(
        df_prop,
        names='Status',
        values='Total',
        hole=0.6,
        color='Status',
        color_discrete_map=CORES
    )
    fig_pie.update_traces(textinfo='percent+label', pull=[0, 0.1])
    fig_pie.update_layout(
        template='plotly_white',
        height=320,
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=30)
    )
    st.plotly_chart(fig_pie, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 5. SIMULADOR DE MACHINE LEARNING (THRESHOLD TUNING)
# -------------------------------------------------------------
st.markdown('<div class="question-card">', unsafe_allow_html=True)
st.markdown('<div class="question-title">Simulador de Limiar de Decisão Antifraude (Threshold Tuning)</div>', unsafe_allow_html=True)
st.write("Avalie em tempo real o trade-off entre bloquear compras de clientes (Falsos Positivos) e deixar fraudes passarem (Falsos Negativos):")

score_risco = (
    -0.45 * df['V14'] 
    - 0.35 * df['V17'] 
    - 0.30 * df['V12'] 
    + 0.25 * df['V4'] 
    + 0.20 * df['V11']
)
df['Score_Fraude'] = 1 / (1 + np.exp(-score_risco))

col_sim_ctrl, col_sim_res = st.columns([5, 7])

with col_sim_ctrl:
    threshold = st.slider(
        "Limiar de Bloqueio Automático:",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
        help="Transações com score acima desse valor serão classificadas como Fraude e bloqueadas."
    )

    y_real = df['Class']
    y_pred = (df['Score_Fraude'] >= threshold).astype(int)

    prec = precision_score(y_real, y_pred, zero_division=0)
    rec = recall_score(y_real, y_pred, zero_division=0)
    f1 = f1_score(y_real, y_pred, zero_division=0)
    cm = confusion_matrix(y_real, y_pred)

    tn, fp, fn, tp = cm.ravel()

    st.markdown(f"""
    * **Precisão:** `{prec:.2%}` *(Confiabilidade dos bloqueios efetuados)*
    * **Recall (Sensibilidade):** `{rec:.2%}` *(Percentual de fraudes capturadas)*
    * **F1-Score:** `{f1:.4f}`
    """)

with col_sim_res:
    z_text = [[f"VN: {tn:,}", f"FP: {fp:,}"], [f"FN: {fn:,}", f"TP: {tp:,}"]]
    fig_cm = go.Figure(data=go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=['Pred: Legítima', 'Pred: Fraude (Bloqueio)'],
        y=['Real: Legítima', 'Real: Fraude'],
        text=z_text,
        texttemplate="%{text}",
        textfont={"size": 14},
        colorscale='Blues',
        showscale=False
    ))
    fig_cm.update_layout(
        template='plotly_white',
        height=280,
        margin=dict(l=40, r=40, t=10, b=30)
    )
    st.plotly_chart(fig_cm, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)