import numpy as np
import pandas as pd
import cvxpy as cp
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
            # self.decision_matrix = pd.DataFrame(self.decision_matrix, columns=columns)

            return self.decision_matrix
        
        except Exception as e:
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
            print(f"An error occurred: {e}")

    def infer_parameters(self):
        try:
            # Define o número de critérios e alternativas de referência
            n_criteria = self.decision_matrix.shape[1]
            n_reference = len(self.reference_set)
            
            # Cria um dicionário com o tamanho das classes para ponderar a função objetivo
            class_sizes = {class_: sum(ref[1] == class_ for ref in self.reference_set) for class_ in set(ref[1] for ref in self.reference_set)}
    
            # Inicializa as variáveis de decisão: pesos e perfis de limite
            weights = cp.Variable(n_criteria, nonneg=True)
            boundary_profiles = cp.Variable((n_reference, n_criteria))
            sigma_plus = cp.Variable(n_reference, nonneg=True)
            sigma_minus = cp.Variable(n_reference, nonneg=True)
    
            # Define a função objetivo
            objective_terms = []
            for i, ref in enumerate(self.reference_set):
                class_size = class_sizes[ref[1]]
                objective_terms.append(sigma_plus[i] / class_size)
                objective_terms.append(sigma_minus[i] / class_size)
            objective = cp.Minimize(cp.sum(objective_terms))
    
            # Define as restrições
            constraints = [cp.sum(weights) == 1]
            
            # Agora definimos ref_classes
            ref_classes = np.array([ref[1] for ref in self.reference_set])
    
            # Adiciona restrições de classificação para as alternativas de referência
            for i, ref in enumerate(self.reference_set):
                ref_value = self.decision_matrix.iloc[i].values
                class_index = np.where(ref_classes == ref[1])[0][0]
                if ref[1] == 'C1':
                    constraints.append(
                        weights @ ref_value - boundary_profiles[class_index, :] <= sigma_plus[i]
                    )
                elif ref[1] == 'C3':
                    constraints.append(
                        boundary_profiles[class_index, :] - weights @ ref_value <= sigma_minus[i]
                    )
                else:
                    # Para classes intermediárias C2, ..., Cq-1
                    constraints.append(
                        weights @ ref_value - boundary_profiles[class_index, :] <= sigma_plus[i]
                    )
                    constraints.append(
                        boundary_profiles[class_index - 1, :] - weights @ ref_value <= sigma_minus[i]
                    )
    
            # Adiciona restrições de monotonicidade para os perfis de limite
            for j in range(n_criteria):
                for i in range(n_reference - 1):
                    constraints.append(
                        boundary_profiles[i, j] >= boundary_profiles[i + 1, j]
                    )
    
            # Resolve o problema de otimização
            problem = cp.Problem(objective, constraints)
            problem.solve()
    
            # Armazena e retorna os pesos e perfis de limite inferidos
            self.weights = weights.value
            self.profiles = boundary_profiles.value
            
            # Gera DataFrames para os resultados
            weights_df = pd.DataFrame(self.weights, columns=['Peso'])
            profiles_df = pd.DataFrame(self.profiles, columns=[f'Perfil {i+1}' for i in range(n_criteria)])
            
            return weights_df, profiles_df
        
        except Exception as e:
            print(f"An error occurred: {e}")


    def calculate_complete_decision_matrix(self, decision_matrix, boundary_profiles, domain):
        try:
            '''
            Step 6.1: Create the Complete Decision Matrix by concatenating the decision matrix,
            the boundary profiles, and the domain.
            '''
            self.complete_decision_matrix = np.vstack((decision_matrix, boundary_profiles, domain))
            
            return self.complete_decision_matrix
        
        except Exception as e:
            print(f"An error occurred: {e}")

    def normalize_decision_matrix(self):
        try:
            '''
            Step 6.2: Normalize the Complete Decision Matrix.
            '''
            self.normalized_decision_matrix = self.complete_decision_matrix / self.complete_decision_matrix.max(axis=0)
            
            return self.normalized_decision_matrix
        
        except Exception as e:
            print(f"An error occurred: {e}")

    def calculate_weighted_normalized_decision_matrix(self, weights):
        try:
            '''
            Step 6.3: Calculate the weighted and normalized Decision Matrix.
            '''
            self.weighted_normalized_decision_matrix = self.normalized_decision_matrix * weights
           
            return self.weighted_normalized_decision_matrix
        
        except Exception as e:
            print(f"An error occurred: {e}")

    def determine_ideal_and_anti_ideal_solutions(self):
        try:
            '''
            Step 6.4: Determine the ideal and anti-ideal solutions.
            '''
            self.v_star = np.max(self.weighted_normalized_decision_matrix, axis=0)
            self.v_minus = np.min(self.weighted_normalized_decision_matrix, axis=0)
            
            return [self.v_star, self.v_minus]
        
        except Exception as e:
            print(f"An error occurred: {e}")


    def calculate_distances(self):
        try:
            '''
            Step 6.5: Calculate the Euclidean distances from each alternative and profile to the ideal and anti-ideal solutions.
            '''
            # Inicializar as listas para armazenar as distâncias
            self.distances_to_ideal = []
            self.distances_to_anti_ideal = []
            self.distances_to_ideal_profiles = []
            self.distances_to_anti_ideal_profiles = []

            # Calcular as distâncias Euclidianas para cada alternativa (a_i)
            for i in range(self.m):  # m é o número de alternativas
                distance_to_ideal = np.sqrt(np.sum((self.weighted_normalized_decision_matrix[i, :] - self.v_star) ** 2))
                distance_to_anti_ideal = np.sqrt(np.sum((self.weighted_normalized_decision_matrix[i, :] - self.v_minus) ** 2))
                self.distances_to_ideal.append(distance_to_ideal)
                self.distances_to_anti_ideal.append(distance_to_anti_ideal)

            # Calcular as distâncias Euclidianas para cada perfil (P_k)
            for k in range(self.q - 1):  # q é o número de perfis + 1
                profile_index = k + self.m  # Perfis são indexados após as alternativas na matriz completa
                distance_to_ideal_profile = np.sqrt(np.sum((self.weighted_normalized_decision_matrix[profile_index, :] - self.v_star) ** 2))
                distance_to_anti_ideal_profile = np.sqrt(np.sum((self.weighted_normalized_decision_matrix[profile_index, :] - self.v_minus) ** 2))
                self.distances_to_ideal_profiles.append(distance_to_ideal_profile)
                self.distances_to_anti_ideal_profiles.append(distance_to_anti_ideal_profile)

            return self.distances_to_ideal, self.distances_to_anti_ideal, self.distances_to_ideal_profiles, self.distances_to_anti_ideal_profiles

        except Exception as e:
            print(f"An error occurred: {e}")

    def calculate_closeness_coefficients(self):
        try:
            '''
            Step 6.6: Calculate the closeness coefficients of each alternative and profile.
            '''
            # Inicializar as listas para armazenar os coeficientes de proximidade
            self.closeness_coefficients = []
            self.profiles_closeness_coefficients = []

            # Calcular os coeficientes de proximidade para cada alternativa
            for i in range(self.m):  # m é o número de alternativas
                closeness_coefficient = self.distances_to_anti_ideal[i] / (self.distances_to_ideal[i] + self.distances_to_anti_ideal[i])
                self.closeness_coefficients.append(closeness_coefficient)

            # Calcular os coeficientes de proximidade para cada perfil
            for k in range(self.q - 1):  # q é o número de perfis + 1
                profile_closeness_coefficient = self.distances_to_anti_ideal_profiles[k] / (self.distances_to_ideal_profiles[k] + self.distances_to_anti_ideal_profiles[k])
                self.profiles_closeness_coefficients.append(profile_closeness_coefficient)

            return self.closeness_coefficients, self.profiles_closeness_coefficients
        
        except Exception as e:
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
            print(f"An error occurred: {e}")

    def sensitivity_analysis(self):
        # placeholder para análise de sensibilidade
        pass
