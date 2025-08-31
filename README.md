#Visão Geral#
Este projeto tem como objetivo desenvolver, avaliar e comparar modelos de credit scoring (análise de risco de crédito) para prever a probabilidade de um cliente se tornar um "mau pagador". Utilizando técnicas de pré-processamento de dados, análise exploratória e modelagem preditiva, o projeto busca identificar os principais fatores de risco e construir um modelo robusto para auxiliar na tomada de decisões financeiras.

O pipeline de desenvolvimento do projeto inclui as seguintes etapas:

Análise Exploratória de Dados (EDA): Entendimento inicial do dataset, distribuição das variáveis e relacionamento com a variável resposta (mau).

Pré-processamento e Feature Engineering: Tratamento de valores ausentes, remoção de outliers, discretização de variáveis numéricas e criação de novas features (agrupamentos).

Análise de Poder Preditivo: Cálculo do Information Value (IV) e Weight of Evidence (WOE) para selecionar as variáveis mais relevantes para a modelagem.

Modelagem de Regressão Logística: Construção e avaliação de um modelo de regressão logística, uma técnica tradicional e interpretável para credit scoring.

Avaliação do Modelo: Análise de métricas de desempenho como AUC, Gini, KS e Acurácia, tanto na base de treinamento quanto em uma base Out-of-Time (OOT) para garantir a estabilidade do modelo.

Modelagem Automatizada com PyCaret: Utilização da biblioteca PyCaret para automatizar o processo de seleção e tunagem de modelos, comparando a performance do modelo manual com outros modelos de machine learning (ex: LightGBM).# Avaliacao_de_Risco_de_Credito_e_-apresentacao_Streamlit

##Tecnologias e Bibliotecas Utilizadas##

pandas: Manipulação e análise de dados.

numpy: Computação numérica.

matplotlib & seaborn: Visualização de dados.

statsmodels: Construção do modelo de Regressão Logística.

scikit-learn: Métricas de avaliação e pré-processamento.

pickle: Serialização e salvamento do modelo.

pycaret: Automação e otimização do fluxo de trabalho de machine learning.

Streamlit: http://localhost:8501/
[streamlit-st-2025-08-31-13-08-21.webm](https://github.com/user-attachments/assets/66b586cf-792d-40b8-af13-f96da2e0d8f5)
