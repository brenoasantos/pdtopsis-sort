import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import os
import time

import main

st.set_page_config(
    page_title='PDTOPSIS-Sort',
    page_icon='⚙️',
    layout='wide',
    initial_sidebar_state="collapsed",
    menu_items={
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "https://github.com/brenoasantos/pdtopsis-sort.git"
    }
)

# create the 'app_input' folder if it doesn't exist
input_folder_path = 'app_input/'

if not os.path.exists(input_folder_path):
    os.makedirs(input_folder_path)
    # st.info(f'Created folder: {input_folder_path}')

st.title('PDTOPSIS-Sort')

uploaded_files = st.file_uploader('Choose a CSV file', accept_multiple_files=True)

for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()

    # # generate a unique filename to avoid overwriting
    # filename = os.path.splitext(uploaded_file.name)[0]  # extract filename without extension
    # extension = os.path.splitext(uploaded_file.name)[1]  # get file extension
    # unique_filename = f'{filename}_{int(time.time())}{extension}'  # add timestamp for uniqueness

    # save the uploaded file to the 'app_input' folder
    with open(os.path.join(input_folder_path, uploaded_file.name), 'wb') as f:
        f.write(bytes_data)

    st.write(f'File saved: {uploaded_file.name} in {input_folder_path} folder')

# check for files already in the folder
if uploaded_files or os.listdir(input_folder_path):  # show button if files are uploaded or already exist
    if st.button('Run PDTOPSIS-Sort'):
        try:
            pdtopsis_sort = main.main()

            # first step
            st.info('Construindo a matriz de decisão...')

            st.table(pdtopsis_sort.load_data())

            # second step
            st.info('Criando tabela de referências...')

            ref_df = pd.DataFrame(pdtopsis_sort.create_ref_set(), columns=['Alternativa', 'Classe'])
            st.table(ref_df)

            # third step
            st.info('Determinando os domínios...')

            domain_df = pd.DataFrame(pdtopsis_sort.determine_domain())
            st.table(domain_df)

            # fourth step
            st.info('Inferindo pesos e perfis de limite...')

            # pdtopsis_sort.infer_parameters()
            # weights_df = pdtopsis_sort.infer_parameters()[0]
            # profiles_df = pdtopsis_sort.infer_parameters()[1]

            # st.table(weights_df)
            # st.table(profiles_df)
            
            # sixth step
            st.info('Matriz de decisão completa...')

            # criar a matriz de decisão completa concatenando X, P e D
            cdm_df = pd.DataFrame(pdtopsis_sort.calculate_complete_decision_matrix())

            st.table(cdm_df)

            # normalizar a matriz de decisão completa
            st.info('Normalizando a matriz de decisão...')

            normalized_dm_df = pd.DataFrame(pdtopsis_sort.normalize_decision_matrix())

            st.table(normalized_dm_df)

            # calcular a matriz de decisão ponderada e normalizada
            pdtopsis_sort.calculate_weighted_normalized_decision_matrix()

            # definir os critérios de benefício
            beneficial_criteria = [i for i in range(pdtopsis_sort.decision_matrix.shape[1])]

            st.info('Determinando as soluções ideal e anti-ideal')

            st.table(pdtopsis_sort.determine_ideal_and_anti_ideal_solutions()[0])
            st.table(pdtopsis_sort.determine_ideal_and_anti_ideal_solutions()[1])


            # função para calcular as distâncias
            distances_results = pdtopsis_sort.calculate_distances()
            distances_to_ideal = distances_results[0]
            distances_to_anti_ideal = distances_results[1]
            distances_to_ideal_profiles = distances_results[2]
            distances_to_anti_ideal_profiles = distances_results[3]

            # converter as listas de distâncias em DataFrames do Pandas
            df_distances_to_ideal = pd.DataFrame(distances_to_ideal, columns=['Distância à Solução Ideal'])
            df_distances_to_anti_ideal = pd.DataFrame(distances_to_anti_ideal, columns=['Distância à Solução Anti-Ideal'])
            df_distances_to_ideal_profiles = pd.DataFrame(distances_to_ideal_profiles, columns=['Distância dos Perfis à Solução Ideal'])
            df_distances_to_anti_ideal_profiles = pd.DataFrame(distances_to_anti_ideal_profiles, columns=['Distância dos Perfis à Solução Anti-Ideal'])

            # mostrar os DataFrames como tabelas no Streamlit
            st.info('Distâncias das alternativas para a solução ideal:')
            st.table(df_distances_to_ideal)

            st.info('Distâncias das alternativas para a solução anti-ideal:')
            st.table(df_distances_to_anti_ideal)

            st.info('Distâncias dos perfis para a solução ideal:')
            st.table(df_distances_to_ideal_profiles)

            st.info('Distâncias dos perfis para a solução anti-ideal:')
            st.table(df_distances_to_anti_ideal_profiles)

            # chamar a função para calcular os coeficientes de proximidade
            closeness_coefficients, profiles_closeness_coefficients = pdtopsis_sort.calculate_closeness_coefficients()

            # converter as listas de coeficientes de proximidade em DataFrames do Pandas
            df_closeness_coefficients = pd.DataFrame(closeness_coefficients, columns=['Coeficiente de Proximidade das Alternativas'])
            df_profiles_closeness_coefficients = pd.DataFrame(profiles_closeness_coefficients, columns=['Coeficiente de Proximidade dos Perfis'])

            # mostrar os DataFrames como tabelas no Streamlit
            st.info('Coeficientes de proximidade das alternativas:')
            st.table(df_closeness_coefficients)

            st.info('Coeficientes de proximidade dos perfis:')
            st.table(df_profiles_closeness_coefficients)

            st.info('Classificando as alternativas...')

            # Chamar a função para classificar as alternativas
            classifications = pdtopsis_sort.classify_alternatives()

            # Preparar os dados para exibir na tabela mantendo a ordem original das alternativas
            alternatives_data = [{
                'Alternative': f'a{i+1}',
                'CI(a)': pdtopsis_sort.closeness_coefficients[i],
                'Sorting': classifications[i]
            } for i in range(len(classifications))]

            # Converter os dados das alternativas em um DataFrame do Pandas
            df_alternatives = pd.DataFrame(alternatives_data)

            # Mostrar a tabela de classificações no Streamlit
            st.info('Resultados da Classificação das Alternativas:')
            st.table(df_alternatives)


            if pdtopsis_sort.errors == 0:
                st.success('PDTOPSIS-Sort executed successfully!')

            else:
                st.error('PDTOPSIS-Sort could not be executed properly.')

        except Exception as e:
            st.error(f'An error occurred: {e}')

# for file in os.listdir(input_folder_path):
#     path = input_folder_path+file
#     os.remove(path)

# os.rmdir(input_folder_path)