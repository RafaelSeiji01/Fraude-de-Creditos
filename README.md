# Plataforma Analitica & Monitoramento Antifraude em Cartoes de Credito

Projeto Desenvolvido para a Disciplina: Data Science and Statistical Computing  
Autor: Rafael Seiji  
Instituicao: FIAP (Faculdade de Informatica e Administracao Paulista)

---

## 1. Visao Geral do Projeto

Fraudes em transacoes com cartoes de credito representam prejuizos financeiros significativos para instituicoes bancarias e desgaste na experiencia dos clientes por meio de bloqueios indevidos (falsos positivos).

Este projeto consiste em uma aplicacao analitica interativa desenvolvida em Python e Streamlit, desenhada para simular uma esteira de inteligencia e monitoramento de risco transacional. O painel integra inferencia estatistica parametrica, reducao de dimensionalidade, avaliacao de bases severamente desbalanceadas e um simulador de limiares de decisao preditiva (Threshold Tuning).

---

## 2. Sobre a Base de Dados

A base utilizada reune registros transacionais realizados por titulares de cartoes de credito europeus ao longo de dois dias:

* Volume Total: 284.807 transacoes
* Fraudes Confirmadas: 492 transacoes (0,172% da base total)
* Prejuizo Acumulado: Superior a $ 60.000,00 nos registros observados

### Anonimizacao e Conformidade (Sigilo Bancario e LGPD)
Para preservar o sigilo das operacoes e os dados sensiveis dos titulares:
* V1 a V28: 28 componentes numericos latentes obtidos via Analise de Componentes Principais (PCA), compactando variaveis originais confidenciais como historico de compras, geolocalizacao e metadados de rede.
* Time: Segundos decorridos entre a transacao inicial e cada evento subsequente.
* Amount: Valor monetario da transacao em dolares ($).
* Class: Variavel-alvo binaria (0 = Transacao Legitima, 1 = Fraude Confirmada).

---

## 3. Detalhamento Estatistico e Interpretacao dos Graficos

### 1. Comportamento Financeiro & Intervalo de Confianca da Media (IC 95%)
* Pergunta Orientadora: O ticket medio de compras fraudulentas difere significativamente das transacoes legitimas?
* Formula Aplicada:
  $$\text{IC}_{\text{Média}} = \bar{X} \pm 1.96 \cdot \left(\frac{s}{\sqrt{n}}\right)$$
* Interpretacao Estatistica:
  * Legitimas ($ 88,29 +- $ 0,92): A amostra expressiva (n = 284.315) faz com que o erro padrao no denominador convirja a incerteza proxima de zero.
  * Fraudes ($ 122,21 +- $ 22,68): A margem de erro e substancialmente mais ampla devido a menor densidade amostral (n = 492) associada a uma alta variancia nos valores aplicados nos golpes.

---

### 2. Taxa de Ataque por Faixa de Valor (IC Proporcional 95%)
* Pergunta Orientadora: Em quais faixas de valor o risco de fraude e proporcionalmente mais elevado?
* Formula Aplicada:
  $$\text{IC}_{\text{Proporção}} = \hat{p} \pm 1.96 \cdot \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}$$
* Interpretacao: A distribuicao de ataques varia conforme a magnitude financeira da compra. Em faixas de valor elevado (> $ 1.000), a baixa densidade de observacoes acarreta intervalos de confianca estatisticos mais largos sobre a proporcao de risco estimada.

---

### 3. Distribuicao Temporal no Ciclo Diario (00h as 23h)
* Pergunta Orientadora: Quais janelas de horario concentram maior exposicao a fraudes?
* Interpretacao: Transacoes legitimas apresentam queda acentuada durante a madrugada (01h as 06h). As fraudes mantem ocorrencia constante nesse mesmo periodo, explorando o intervalo de menor atencao e demora de contestacao por parte dos clientes.

---

### 4. Projecao Espacial 2D em Componentes PCA (V14 vs V17)
* Pergunta Orientadora: As variaveis compactadas permitem isolar compras suspeitas?
* Interpretacao: Cada ponto no plano cartesiano expressa uma transacao posicionada pelo seu vetor comportamental. No cruzamento de eixos criticos (como V14 contra V17), evidencia-se separabilidade linear: transacoes fraudulentas concentram-se no quadrante inferior esquerdo (V14 < -5 e V17 < -2), indicando aos modelos preditivos a viabilidade de criacao de regras de bloqueio.

---

### 5. Desbalanceamento Severo de Classes
* Pergunta Orientadora: Por que a acuracia tradicional nao deve ser a metrica orientadora deste modelo?
* Interpretacao: Em bases onde 99,828% das operacoes sao legitimas, um classificador trivial que aprove todas as transacoes obteria 99,83% de acuracia sem capturar nenhuma fraude. A avaliacao do classificador orienta-se, portanto, por Precisao, Recall e F1-Score.

---

### 6. Simulador de Limiar de Decisao (Threshold Tuning)
* Aplicacao: Ferramenta interativa que permite ao analista ajustar a sensibilidade do algoritmo (0.01 a 0.99):
  * Limiares baixos: Maximizam o Recall (captura da maioria dos golpes), aumentando os Falsos Positivos (atrito operacional com clientes legitimos).
  * Limiares altos: Maximizam a Precisao (bloqueios apenas em alta certeza), aumentando os Falsos Negativos (prejuizo financeiro por chargebacks).
* Matriz de Confusao Dinamica: Apresenta em tempo real a distribuicao de Verdadeiros Positivos (TP), Falsos Positivos (FP), Verdadeiros Negativos (TN) e Falsos Negativos (FN).

---

## 4. Engenharia de Dados e Otimizacao de Deploy

Para operar de forma estavel no Streamlit Community Cloud sob limitacoes de memoria RAM e restricoes de upload de arquivos no GitHub:

1. Formato Parquet: O dataset em CSV original (~143 MB) foi convertido para formato colunar Parquet com compressao Snappy (~25 MB), reduzindo drasticamente o tempo de leitura e eliminando gargalos de I/O.
2. Otimizacao de Memoria (Downcasting): Variaveis do tipo float64 foram convertidas para float32 e a classe categorica para int8, contendo a alocacao total em memoria da aplicacao.
3. Cacheamento em Runtime: Uso do decorator @st.cache_data para garantir resposta instantanea nas interacoes de filtros e simulacoes sem recarregar a base.

---

## 5. Instrucoes de Execucao Local

1. Clonar o repositorio:
   git clone https://github.com/RafaelSeiji01/Fraude-de-Creditos.git
   cd Fraude-de-Creditos

2. Criar e ativar o ambiente virtual:
   python -m venv venv
   # No Windows:
   venv\Scripts\activate
   # No Linux/macOS:
   source venv/bin/activate

3. Instalar as dependencias:
   pip install -r requirements.txt

4. Executar a aplicacao:
   streamlit run main.py

---

## 6. Dependencias do Projeto

* streamlit
* pandas
* numpy
* plotly
* scikit-learn
* pyarrow

---

## 7. Autor

* Nome: Rafael Seiji
* Formacao: Graduando em Engenharia de Software — FIAP
* Perfil Profissional: https://www.linkedin.com/in/rafael-seiji-39961b333/
* Repositorio GitHub: https://github.com/RafaelSeiji01/Fraude-de-Creditos
