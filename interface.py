import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import os
import time

import main

# create the 'app_input' folder if it doesn't exist
input_folder_path = 'app_input'

if not os.path.exists(input_folder_path):
    os.makedirs(input_folder_path)
    st.info(f'Created folder: {input_folder_path}')

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

            #first step
            st.write('Construindo a matriz de decisão...')
            time.sleep(1)

            st.table(pdtopsis_sort.load_data())

            #second step
            st.write('Criando tabela de referências...')
            time.sleep(1)

            ref_df = pd.DataFrame(pdtopsis_sort.create_ref_set(), columns=['Alternativa', 'Classe'])
            st.table(ref_df)

            #third step
            st.write('Determinando os domínios...')
            time.sleep(1)

            domain_df = pd.DataFrame(pdtopsis_sort.determine_domain())
            print(pdtopsis_sort.determine_domain())
            st.table(domain_df)

            #fourth step
            st.write('Inferindo pesos e perfis de limite...')
            time.sleep(1)

            pdtopsis_sort.infer_parameters()

            #sixth step
            st.write('Classificando as alternativas...')
            time.sleep(1)

            # criar a matriz de decisão completa concatenando X, P e D
            pdtopsis_sort.calculate_complete_decision_matrix(pdtopsis_sort.decision_matrix.values,
                                            pdtopsis_sort.profiles,
                                            np.array([pdtopsis_sort.domain['ideal'], pdtopsis_sort.domain['anti_ideal']]))

            # normalizar a matriz de decisão completa                                    
            pdtopsis_sort.normalize_decision_matrix()

            # calcular a matriz de decisão ponderada e normalizada
            pdtopsis_sort.calculate_weighted_normalized_decision_matrix(pdtopsis_sort.weights)

            # definir os critérios de benefício
            beneficial_criteria = [i for i in range(pdtopsis_sort.decision_matrix.shape[1])]

            # determinar as soluções ideais e anti-ideais
            pdtopsis_sort.determine_ideal_and_anti_ideal_solutions(beneficial_criteria)

            # calcular as distâncias Euclidianas para cada alternativa e perfil
            pdtopsis_sort.calculate_distances()

            # calcular os coeficientes de proximidade para cada alternativa e perfil
            pdtopsis_sort.calculate_closeness_coefficients()

            # classificar as alternativas com base nos coeficientes de proximidade
            classifications = pdtopsis_sort.classify_alternatives(pdtopsis_sort.profiles_closeness_coefficients)

            # apresentar os resultados finais
            for i, classification in enumerate(classifications):
                print(f'Alternativa {i + 1}: Classe {classification}')

            st.success('PDTOPSIS-Sort executed successfully!')

        except Exception as e:
            st.error(f'An error occurred: {e}')