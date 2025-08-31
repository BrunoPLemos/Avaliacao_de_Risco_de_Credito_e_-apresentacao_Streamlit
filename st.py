import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model # Importa as funções do PyCaret

# 1. Título da Aplicação
st.set_page_config(layout="wide")
st.title("Aplicação de Escoragem de Crédito (PyCaret)")
st.write("Esta aplicação permite carregar um arquivo CSV, pré-processar os dados e gerar uma pontuação de risco de crédito usando um modelo treinado com PyCaret.")

# 2. Carregar o Modelo Salvo pelo PyCaret
try:
    # 'Final data Model 24JAgo2025.pkl'
    modelo_pycaret = load_model('Final data Model 24JAgo2025')
    st.sidebar.success("Modelo PyCaret carregado com sucesso!")
except Exception as e:
    st.sidebar.error(f"Erro: Não foi possível carregar o modelo PyCaret. Certifique-se de que 'Final data Model 24JAgo2025.pkl' está na mesma pasta. Erro: {e}")
    st.stop()

# 3. Carregar CSV no Streamlit
st.sidebar.header("Upload do seu arquivo CSV")
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV", type=["csv"])

df_input = None
if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.sidebar.success("Arquivo CSV carregado com sucesso!")
        # Converte a coluna 'data_ref' para o tipo datetime, que é o que o modelo PyCaret espera.
        df_input['data_ref'] = pd.to_datetime(df_input['data_ref'])
        st.subheader("Prévia dos Dados Carregados:")
        st.write(df_input.head())

        # 4. Escoragem da Base usando o modelo PyCaret 
        st.subheader("Gerando Escores de Risco...")
        # O PyCaret lida com todas as transformações de pré-processamento que foram definidas no `setup`
        # e aplicadas no `predict_model` durante o treinamento.
        df_escorado = predict_model(modelo_pycaret, data=df_input)

        
        st.success("Escoragem realizada com sucesso!")

        st.subheader("Resultados da Escoragem:")
        st.write(df_escorado.head())

        # Exibir o DataFrame completo com os resultados
        if st.checkbox("Mostrar DataFrame completo com escoragem"):
            st.dataframe(df_escorado)

        # Opção para download dos resultados
        csv_output = df_escorado.to_csv(index=False)
        st.download_button(
            label="Baixar resultados escorados (CSV)",
            data=csv_output,
            file_name="dados_escorados_pycaret.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Ocorreu um erro durante a leitura do CSV ou escoragem: {e}")
        st.exception(e)
else:
    st.info("Por favor, faça o upload de um arquivo CSV para continuar.")
    st.stop()