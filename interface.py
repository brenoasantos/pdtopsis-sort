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


            # calcular as distâncias Euclidianas para cada alternativa e perfil
            pdtopsis_sort.calculate_distances()
            st.table(pdtopsis_sort.calculate_distances()[0])
            st.table(pdtopsis_sort.calculate_distances()[1])


            # calcular os coeficientes de proximidade para cada alternativa e perfil
            # pdtopsis_sort.calculate_closeness_coefficients()

            st.info('Classificando as alternativas...')

            # st.write(pdtopsis_sort.classify_alternatives())

            # # apresentar os resultados finais
            # for i, classification in enumerate(classifications):
            #     print(f'Alternativa {i + 1}: Classe {classification}')

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