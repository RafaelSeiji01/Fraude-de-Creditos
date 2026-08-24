import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Estatísticas & Modelagem", layout="wide")

st.markdown("""
    <style>
        .stApp { background-color: #f4f6f9; }
        .dashboard-card {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 22px 24px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
            margin-bottom: 20px;
        }
        .card-heading {
            font-size: 1.1rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 4px;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def carregar_dados():
    diretorio_pagina = os.path.dirname(os.path.abspath(__file__))
    diretorio_raiz = os.path.abspath(os.path.join(diretorio_pagina, '..'))
    diretorio_trabalho = os.getcwd()

    # Mapeamento de possíveis diretórios onde o CSV/ZIP pode estar localizado
    caminhos = [
        os.path.join(diretorio_raiz, 'archive', 'creditcard.csv'),
        os.path.join(diretorio_raiz, 'archive', 'creditcard.zip'),
        os.path.join(diretorio_trabalho, 'archive', 'creditcard.csv'),
        os.path.join(diretorio_trabalho, 'archive', 'creditcard.zip'),
        os.path.join(diretorio_raiz, 'creditcard.csv'),
        os.path.join(diretorio_raiz, 'creditcard.zip'),
        os.path.join(diretorio_trabalho, 'creditcard.csv'),
        os.path.join(diretorio_trabalho, 'creditcard.zip'),
        os.path.join(diretorio_pagina, 'archive', 'creditcard.csv'),
    ]

    caminho_csv = None
    for c in caminhos:
        if os.path.exists(c):
            caminho_csv = c
            break

    if caminho_csv is None:
        st.error(f"Arquivo creditcard (CSV ou ZIP) não foi encontrado! Buscado em: {diretorio_raiz}")
        st.stop()

    is_zip = caminho_csv.endswith('.zip')
    df = pd.read_csv(caminho_csv, compression='zip' if is_zip else None)
    
    df['Status'] = df['Class'].map({0: 'Legítima', 1: 'Fraude'})
    df['Hora_do_Dia'] = ((df['Time'] / 3600) % 24).astype(int)
    
    bins_valor = [-1, 10, 50, 200, 500, 1000, float('inf')]
    labels_valor = ['$0 - $10', '$10 - $50', '$50 - $200', '$200 - $500', '$500 - $1.000', '> $1.000']
    df['Faixa_Valor'] = pd.cut(df['Amount'], bins=bins_valor, labels=labels_valor)
    return df

df = carregar_dados()
CORES = {'Legítima': '#2563eb', 'Fraude': '#dc2626'}

st.title("Inferência Estatística & Machine Learning")
st.caption("Validação com Intervalos de Confiança (95%) e Ajuste de Limiar Antifraude")

tab_ic, tab_ml = st.tabs(["Intervalos de Confiança (95%)", "Simulador de Machine Learning"])

with tab_ic:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">Ticket Médio com Margem de Erro Paramétrica (IC 95%)</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 7])
    
    valores_leg = df[df['Class'] == 0]['Amount']
    valores_frd = df[df['Class'] == 1]['Amount']
    m_leg, erro_leg = valores_leg.mean(), 1.96 * (valores_leg.std() / np.sqrt(len(valores_leg)))
    m_frd, erro_frd = valores_frd.mean(), 1.96 * (valores_frd.std() / np.sqrt(len(valores_frd)))

    with col1:
        st.latex(r"\text{IC}_{\text{Média}} = \bar{X} \pm 1.96 \cdot \left(\frac{s}{\sqrt{n}}\right)")
        st.markdown(f"""
        * **Legítima (n = {len(valores_leg):,}):** Média **${m_leg:.2f}** (±${erro_leg:.2f})
        * **Fraude (n = {len(valores_frd):,}):** Média **${m_frd:.2f}** (±${erro_frd:.2f})
        """)
        st.info("A margem da fraude é maior devido ao tamanho amostral reduzido e à alta variabilidade dos gastos.")

    with col2:
        fig_ic = go.Figure()
        fig_ic.add_trace(go.Bar(
            x=['Legítima', 'Fraude'],
            y=[m_leg, m_frd],
            error_y=dict(type='data', array=[erro_leg, erro_frd], visible=True, thickness=2),
            marker_color=['#2563eb', '#dc2626'],
            text=[f"${m_leg:.2f}<br>±${erro_leg:.2f}", f"${m_frd:.2f}<br>±${erro_frd:.2f}"],
            textposition='outside'
        ))
        fig_ic.update_layout(template='plotly_white', height=300, yaxis=dict(range=[0, (m_frd + erro_frd) * 1.35]), margin=dict(l=30, r=30, t=10, b=30))
        st.plotly_chart(fig_ic, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_ml:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">Tuning de Limiar (Threshold) & Matriz de Confusão</div>', unsafe_allow_html=True)
    
    score_risco = -0.45 * df['V14'] - 0.35 * df['V17'] - 0.30 * df['V12'] + 0.25 * df['V4'] + 0.20 * df['V11']
    df['Score'] = 1 / (1 + np.exp(-score_risco))
    
    c_m1, c_m2 = st.columns([5, 7])
    with c_m1:
        th = st.slider("Limiar de Corte:", 0.01, 0.99, 0.50, 0.01)
        y_pred = (df['Score'] >= th).astype(int)
        
        prec = precision_score(df['Class'], y_pred, zero_division=0)
        rec = recall_score(df['Class'], y_pred, zero_division=0)
        f1 = f1_score(df['Class'], y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(df['Class'], y_pred).ravel()
        
        st.markdown(f"* **Precisão:** `{prec:.2%}`\n* **Recall:** `{rec:.2%}`\n* **F1-Score:** `{f1:.4f}`")
    
    with c_m2:
        fig_cm = go.Figure(data=go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=['Pred: Legítima', 'Pred: Fraude'],
            y=['Real: Legítima', 'Real: Fraude'],
            text=[[f"VN: {tn:,}", f"FP: {fp:,}"], [f"FN: {fn:,}", f"TP: {tp:,}"]],
            texttemplate="%{text}",
            colorscale='Blues',
            showscale=False
        ))
        fig_cm.update_layout(template='plotly_white', height=260, margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)