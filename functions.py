import numpy as np
import pandas as pd
# import cvxpy as cp
import math

class PDTOPSIS_Sort:
    def __init__(self, matrix_values_filepath, input_filepath):
        self.matrix_values_csv = matrix_values_filepath
        self.input_csv = input_filepath
        self.decision_matrix = None
        self.reference_set = None
        self.domain = None
        self.weights = None
        self.profiles = None
        self.errors = 0

        # load data
        self.load_data()

    def load_data(self):
        try:          
            # carrega os dados do arquivo CSV com inputs para um dataframe
            self.inputs = pd.read_csv(self.input_csv, delimiter=';')
            
            # carrega os dados do arquivo CSV com valores para a matriz de decisão (dataframe)
            columns = []
            columns.append(value for value in self.inputs.crit_code.tolist() if value != ' ')
            
            self.decision_matrix = pd.read_csv(self.matrix_values_csv, delimiter=';', header=None)
            self.list_decision_matrix = pd.read_csv(self.matrix_values_csv, delimiter=';', header=None).values.tolist()
            # self.decision_matrix = pd.DataFrame(self.decision_matrix, columns=columns)

            return self.decision_matrix
        
        except Exception as e:
            self.errors = self.errors+1
            print(f"An error occurred: {e}")

    def create_ref_set(self):
        try:
            self.reference_set = []

            # cria/carrega a matrix contendo alternativas de referências e suas classes
            for index, row in self.inputs.iterrows():
                if row.ref_alternatives != " ":
                    self.reference_set.append([row.ref_alternatives, row.ref_alt_class])

            return self.reference_set
        
        except Exception as e:
            self.errors = self.errors+1
            print(f"An error occurred: {e}")

    def determine_domain(self):
        try:
            self.domain = {
                'ideal': [],
                'anti_ideal': []
            }

            # carrega o dicionário com valores máximos e mínimos de cada critério
            for label, content in self.decision_matrix.items():
                self.domain['ideal'].append(max(content))
                self.domain['anti_ideal'].append(min(content))
            
            return self.domain

        except Exception as e:
            self.errors = self.errors+1
            print(f"An error occurred: {e}")

    # def infer_parameters(self):
    #     try:
    #         # Define o número de critérios e alternativas de referência
    #         n_criteria = self.decision_matrix.shape[1]
    #         n_reference = len(self.reference_set)

    #         print(n_criteria)
    #         print(n_reference)
            
    #         # Cria um dicionário com o tamanho das classes para ponderar a função objetivo
    #         class_sizes = {class_: sum(ref[1] == class_ for ref in self.reference_set) for class_ in set(ref[1] for ref in self.reference_set)}
    
    #         print(class_sizes)

    #         # Inicializa as variáveis de decisão: pesos e perfis de limite
    #         weights = cp.Variable(n_criteria, nonneg=True)
    #         boundary_profiles = cp.Variable((n_reference, n_criteria))
    #         sigma_plus = cp.Variable(n_reference, nonneg=True)
    #         sigma_minus = cp.Variable(n_reference, nonneg=True)

    #         print(weights)
    #         print(boundary_profiles)
    #         print(sigma_plus)
    #         print(sigma_minus)
    
    #         # Define a função objetivo
    #         objective_terms = []

    #         for i, ref in enumerate(self.reference_set):
    #             class_size = class_sizes[ref[1]]
    #             objective_terms.append(sigma_plus[i] / class_size)
    #             objective_terms.append(sigma_minus[i] / class_size)

    #         objective = cp.Minimize(cp.sum(objective_terms))

    #         print(objective)
    
    #         # Define as restrições
    #         constraints = [cp.sum(weights) == 1]

    #         print(constraints)
            
    #         # Agora definimos ref_classes
    #         ref_classes = np.array([ref[1] for ref in self.reference_set])

    #         print(ref_classes)
    
    #         # Adiciona restrições de classificação para as alternativas de referência
    #         for i, ref in enumerate(self.reference_set):
    #             ref_value = self.decision_matrix.iloc[i].values
    #             class_index = np.where(ref_classes == ref[1])[0][0]

    #             if ref_value.ndim == 1:
    #                 ref_value = ref_value[:, np.newaxis]  # Convert to a column vector if it's not already

    #             if ref[1] == 'C1':
    #                 constraints.append(cp.sum(cp.multiply(weights, ref_value)) - boundary_profiles[class_index, :] <= sigma_plus[i])

    #             elif ref[1] == 'C3':
    #                 constraints.append(boundary_profiles[class_index, :] - cp.sum(cp.multiply(weights, ref_value)) <= sigma_minus[i])
                
    #             else:

    #                 # Para classes intermediárias C2, ..., Cq-1
    #                 constraints.append(cp.sum(cp.multiply(weights, ref_value)) - boundary_profiles[class_index, :] <= sigma_plus[i])
    #                 constraints.append(boundary_profiles[class_index - 1, :] - cp.sum(cp.multiply(weights, ref_value)) <= sigma_minus[i])
    
    #         # Adiciona restrições de monotonicidade para os perfis de limite
    #         for j in range(n_criteria):
    #             for i in range(n_reference - 1):
    #                 constraints.append(
    #                     boundary_profiles[i, j] >= boundary_profiles[i + 1, j]
    #                 )
            
    #         print(constraints)
    
    #         # Resolve o problema de otimização
    #         problem = cp.Problem(objective, constraints)
    #         problem.solve()
    
    #         # Armazena e retorna os pesos e perfis de limite inferidos
    #         self.weights = weights.value
    #         self.profiles = boundary_profiles.value

    #         print(self.weights)
    #         print(self.profiles)
            
    #         # Gera DataFrames para os resultados
    #         weights_df = pd.DataFrame(self.weights, columns=[f'G {i+1}' for i in range(n_criteria)])
    #         profiles_df = pd.DataFrame(self.profiles, columns=[f'G {i+1}' for i in range(n_criteria)])
            
    #         print(weights_df)
    #         print(profiles_df)

    #         return weights_df, profiles_df
        
    #     except Exception as e:
    #          self.errors = self.errors+1
    #         print(f"An error occurred: {e}")

    def calculate_complete_decision_matrix(self):
        self.complete_decision_matrix = self.list_decision_matrix

        profiles = [[0.0816, 0.0616], [0.1174, 0.0675], [0.2089, -0.0047], [0.3669, 0.2458], [1.4659, 0.9165], [0.7805, 0.4404], [0.6375, 0.1256], [1.0971, 0.9865], [2.0729, 2.5155], [2.4333, -0.2015]]
        domain = [[i for i in self.domain['ideal']], [i for i in self.domain['anti_ideal']]]
        
        try:
            '''
            Step 6.1: Create the Complete Decision Matrix by concatenating the decision matrix,
            the boundary profiles, and the domain.
            '''
            row1 = []
            row2 = []

            for row in profiles:
                row1.append(row[0])
                row2.append(row[1])

            self.complete_decision_matrix.append(row1)
            self.complete_decision_matrix.append(row2)
            self.complete_decision_matrix.append(domain[0])
            self.complete_decision_matrix.append(domain[1])

            return self.complete_decision_matrix
        
        except Exception as e:
            self.errors = self.errors+1
            print(f"An error occurred: {e}")

    def normalize_decision_matrix(self):
        try:
            '''
            Step 6.2: Normalize the Complete Decision Matrix.
            '''
            np_matrix = np.array(self.complete_decision_matrix)
            max_values = np.max(np_matrix, axis=0)

            self.normalized_matrix = np_matrix/max_values

            return self.normalized_matrix
            
        except Exception as e:
            self.errors = self.errors+1
            print(f"An error occurred: {e}")

    def calculate_weighted_normalized_decision_matrix(self):
        weights = [0.0079, 0.0099, 0.0170, 0.1187, 0.1664, 0.1486, 0.2382,  0.1292, 2.1462, 0.0180]

        try:
            '''
            Step 6.3: Calculate the weighted and normalized Decision Matrix.
            '''
            self.weighted_normalized_matrix = []

            # Itere sobre cada linha da matriz normalizada
            for row in self.normalized_matrix:
            # Multiplique cada elemento (valor de critério) pelo peso correspondente
                weighted_row = [value*weights[i] for i, value in enumerate(row)]

                # Adicione a linha ponderada à nova lista de resultados
                self.weighted_normalized_matrix.append(weighted_row)

            self.row_size = len(self.weighted_normalized_matrix)
            self.column_size = len(self.weighted_normalized_matrix[0])

            return self.weighted_normalized_matrix
        
        except Exception as e:
            self.errors = self.errors+1
            print(f"An error occurred: {e}")

    def determine_ideal_and_anti_ideal_solutions(self):
        try:
            '''
            Step 6.4: Determine the ideal and anti-ideal solutions.
            '''
            self.v_star = np.max(self.weighted_normalized_matrix, axis=0)
            self.v_minus = np.min(self.weighted_normalized_matrix, axis=0)

            # Transformando a matriz em uma linha com uma única coluna
            v_star_reshaped = self.v_star.reshape(-1, 1)
            v_minus_reshaped = self.v_minus.reshape(-1, 1)

            # Transpondo a matriz para obter uma linha com múltiplas colunas
            self.v_star = v_star_reshaped.T
            self.v_minus = v_minus_reshaped.T

            # self.combined_df.reset_index(drop=True, inplace=True)
            return self.v_star, self.v_minus
        
        except Exception as e:
            self.errors = self.errors+1
            print(f"An error occurred: {e}")


    def calculate_distances(self):
        try:
            self.weighted_normalized_matrix = np.array(self.weighted_normalized_matrix)
            self.v_star = np.array(self.v_star)
            self.v_minus = np.array(self.v_minus)
            
            # Inicializa as listas para as distâncias
            self.distances_to_ideal = []
            self.distances_to_anti_ideal = []

            # Calcula as distâncias para as alternativas
            for i in range(m):  # m é o número de alternativas
                distance_to_ideal = np.sqrt(np.sum((self.weighted_normalized_matrix[i, :] - self.v_star) ** 2))
                distance_to_anti_ideal = np.sqrt(np.sum((self.weighted_normalized_matrix[i, :] - self.v_minus) ** 2))
                self.distances_to_ideal.append(distance_to_ideal)
                self.distances_to_anti_ideal.append(distance_to_anti_ideal)

            # Calcula as distâncias para os perfis
            self.distances_to_ideal_profiles = []
            self.distances_to_anti_ideal_profiles = []
            for k in range(1, q):  # q é o número de perfis + 1
                profile_index = k + m - 1  # Índice do perfil na matriz normalizada
                distance_to_ideal_profile = np.sqrt(np.sum((self.weighted_normalized_matrix[profile_index, :] - self.v_star) ** 2))
                distance_to_anti_ideal_profile = np.sqrt(np.sum((self.weighted_normalized_matrix[profile_index, :] - self.v_minus) ** 2))
                self.distances_to_ideal_profiles.append(distance_to_ideal_profile)
                self.distances_to_anti_ideal_profiles.append(distance_to_anti_ideal_profile)
            return (self.distances_to_ideal, self.distances_to_anti_ideal,
                    self.distances_to_ideal_profiles, self.distances_to_anti_ideal_profiles)

        except Exception as e:
            self.errors += 1
            print(f"Ocorreu um erro: {e}")
    def calculate_closeness_coefficients(self):
        try:
            '''
            Step 6.6: Calculate the closeness coefficients of each alternative and profile.
            '''
            # Inicializar as listas para armazenar os coeficientes de proximidade
            self.closeness_coefficients = []
            self.profiles_closeness_coefficients = []

            # Calcular os coeficientes de proximidade para cada alternativa
            for i in range(n_alternatives):  # m é o número de alternativas
                closeness_coefficient = self.distances_to_anti_ideal[i] / (self.distances_to_ideal[i] + self.distances_to_anti_ideal[i])
                self.closeness_coefficients.append(closeness_coefficient)

            # Calcular os coeficientes de proximidade para cada perfil
            for k in range(self.q - 1):  # q é o número de perfis + 1
                profile_closeness_coefficient = self.distances_to_anti_ideal_profiles[k] / (self.distances_to_ideal_profiles[k] + self.distances_to_anti_ideal_profiles[k])
                self.profiles_closeness_coefficients.append(profile_closeness_coefficient)

            return self.closeness_coefficients, self.profiles_closeness_coefficients
        
        except Exception as e:
            self.errors = self.errors+1
            print(f"An error occurred: {e}")

    def classify_alternatives(self):
        try:
            '''
            Step 6.7: Classify the alternatives by making comparisons between their closeness coefficients.
            '''
            # supondo que self.profiles_closeness_coefficients seja uma lista ordenada dos coeficientes de proximidade dos perfis
            # supondo também que self.closeness_coefficients contém os coeficientes de proximidade das alternativas
            self.classifications = []  # Esta lista irá armazenar a classificação de cada alternativa

            # iniciar a classificação
            for closeness_coefficient in self.closeness_coefficients:
                # inicialmente, supõe-se que a alternativa não está classificada em nenhuma classe
                class_assignment = None

                # verificar se a alternativa pertence à melhor classe C1
                if closeness_coefficient >= self.profiles_closeness_coefficients[0]:
                    class_assignment = 'C1'

                # verificar se a alternativa pertence à pior classe Cq
                elif closeness_coefficient < self.profiles_closeness_coefficients[-1]:
                    class_assignment = 'Cq'

                # se não, verificar as classes intermediárias
                else:
                    for k in range(1, len(self.profiles_closeness_coefficients)):
                        if self.profiles_closeness_coefficients[k-1] > closeness_coefficient >= self.profiles_closeness_coefficients[k]:
                            class_assignment = f'C{k+1}'
                            break

                # se, após as verificações, ainda não tiver sido possível classificar a alternativa
                # isso indicaria um erro na lógica ou nos dados
                if class_assignment is None:
                    raise ValueError('Não foi possível classificar a alternativa com base nos coeficientes de proximidade fornecidos.')

                self.classifications.append(class_assignment)

            # retornar as classificações
            return self.classifications
        
        except Exception as e:
            self.errors = self.errors+1
            print(f"An error occurred: {e}")
