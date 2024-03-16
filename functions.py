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
            # carrega os dados do arquivo CSV com valores para a matriz de decisão (dataframe)
            self.decision_matrix = pd.read_csv(self.matrix_values_csv, delimiter=';', header=None)
            
            # carrega os dados do arquivo CSV com inputs para um dataframe
            self.inputs = pd.read_csv(self.input_csv, delimiter=';')
        
        except Exception as e:
            print(f"An error occurred: {e}")

    def create_ref_set(self):
        try:
            self.reference_set = []

            # cria/carrega a matrix contendo alternativas de referências e suas classes
            for index, row in self.inputs.iterrows():
                if row.ref_alternatives != " ":
                    self.reference_set.append([row.ref_alternatives, row.ref_alt_class])

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

        except Exception as e:
            print(f"An error occurred: {e}")

    def infer_parameters(self):
        n_criteria = self.decision_matrix.shape[1]  # número de critérios
        n_alternatives = self.decision_matrix.shape[0]  # número de alternativas
        n_reference = len(self.reference_set)  # número de alternativas de referência
        
        # variáveis de decisão
        weights = cp.Variable(n_criteria, nonneg=True)
        boundary_profiles = cp.Variable((n_reference, n_criteria))
        
        # variáveis de erro para cada alternativa de referência
        sigma_plus = cp.Variable(n_reference, nonneg=True)
        sigma_minus = cp.Variable(n_reference, nonneg=True)

        # obter os valores de classificação das alternativas de referência
        ref_classes = np.array([ref[1] for ref in self.reference_set])

        # função objetivo: Minimizar a soma das variáveis de erro
        objective = cp.Minimize(cp.sum(sigma_plus) + cp.sum(sigma_minus))

        # restrições
        constraints = []

        # restrições de pesos
        constraints.append(cp.sum(weights) == 1)

        # as alternativas de referência devem ser classificadas corretamente
        # assumindo que a classe 'C1' é a melhor e 'Cn' é a pior
        for i, ref in enumerate(self.reference_set):
            ref_value = self.decision_matrix.iloc[i, :]
            class_index = np.where(ref_classes == ref[1])[0][0]
            
            # restrição para a classe superior (benefício)
            if ref[1] == 'C1':
                constraints.append(boundary_profiles[class_index, :] * weights - ref_value <= sigma_plus[i])
            # restrição para a classe inferior (custo)
            elif ref[1] == 'Cn':
                constraints.append(ref_value - boundary_profiles[class_index, :] * weights <= sigma_minus[i])
            # restrições para as classes intermediárias
            else:
                constraints.append(boundary_profiles[class_index, :] * weights - ref_value <= sigma_plus[i])
                constraints.append(ref_value - boundary_profiles[class_index - 1, :] * weights <= sigma_minus[i])
        
        # monotonicidade dos perfis de limite entre classes
        for j in range(n_criteria):
            for k in range(n_reference - 1):
                constraints.append(boundary_profiles[k, j] >= boundary_profiles[k + 1, j])

        # resolver o problema
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # armazenar os resultados
        self.weights = weights.value
        self.profiles = boundary_profiles.value

        print("Inferred weights: ", self.weights)
        print("Inferred boundary profiles: ", self.profiles)

    def calculate_complete_decision_matrix(self, decision_matrix, boundary_profiles, domain):
        '''
        Step 6.1: Create the Complete Decision Matrix by concatenating the decision matrix,
        the boundary profiles, and the domain.
        '''
        self.complete_decision_matrix = np.vstack((decision_matrix, boundary_profiles, domain))

    def normalize_decision_matrix(self):
        '''
        Step 6.2: Normalize the Complete Decision Matrix.
        '''
        self.normalized_decision_matrix = self.complete_decision_matrix / self.complete_decision_matrix.max(axis=0)

    def calculate_weighted_normalized_decision_matrix(self, weights):
        '''
        Step 6.3: Calculate the weighted and normalized Decision Matrix.
        '''
        self.weighted_normalized_decision_matrix = self.normalized_decision_matrix * weights

    def determine_ideal_and_anti_ideal_solutions(self, beneficial_criteria, cost_criteria):
        '''
        Step 6.4: Determine the ideal and anti-ideal solutions.
        '''
        self.v_star = np.max(self.weighted_normalized_decision_matrix[:, beneficial_criteria], axis=0)
        self.v_star = np.concatenate((self.v_star, np.min(self.weighted_normalized_decision_matrix[:, cost_criteria], axis=0)))

        self.v_minus = np.min(self.weighted_normalized_decision_matrix[:, beneficial_criteria], axis=0)
        self.v_minus = np.concatenate((self.v_minus, np.max(self.weighted_normalized_decision_matrix[:, cost_criteria], axis=0)))

    def calculate_distances(self):
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

    def calculate_closeness_coefficients(self):
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


    def classify_alternatives(self, profiles_closeness_coefficients):
        '''
        Step 6.7: Classify the alternatives by making comparisons between their closeness coefficients.
        '''
        # Supondo que self.profiles_closeness_coefficients seja uma lista ordenada dos coeficientes de proximidade dos perfis
        # Supondo também que self.closeness_coefficients contém os coeficientes de proximidade das alternativas
        self.classifications = []  # Esta lista irá armazenar a classificação de cada alternativa

        # Iniciar a classificação
        for closeness_coefficient in self.closeness_coefficients:
            # Inicialmente, supõe-se que a alternativa não está classificada em nenhuma classe
            class_assignment = None

            # Verificar se a alternativa pertence à melhor classe C1
            if closeness_coefficient >= self.profiles_closeness_coefficients[0]:
                class_assignment = 'C1'
            # Verificar se a alternativa pertence à pior classe Cq
            elif closeness_coefficient < self.profiles_closeness_coefficients[-1]:
                class_assignment = 'Cq'
            # Se não, verificar as classes intermediárias
            else:
                for k in range(1, len(self.profiles_closeness_coefficients)):
                    if self.profiles_closeness_coefficients[k-1] > closeness_coefficient >= self.profiles_closeness_coefficients[k]:
                        class_assignment = f'C{k+1}'
                        break

            # Se, após as verificações, ainda não tiver sido possível classificar a alternativa
            # isso indicaria um erro na lógica ou nos dados
            if class_assignment is None:
                raise ValueError('Não foi possível classificar a alternativa com base nos coeficientes de proximidade fornecidos.')

            self.classifications.append(class_assignment)

        # Retornar as classificações ou manipulá-las conforme necessário
        return self.classifications
    
    def sensitivity_analysis(self):
        # placeholder para análise de sensibilidade
        pass

    def run_algorithm(self):
        # executar todos os passos do algoritmo sequencialmente
        self.create_ref_set()
        self.determine_domain()
        # self.infer_parameters()
        # self.validate_parameters()
        # self.classify_alternatives()
        # self.sensitivity_analysis()
        # output final: apresentação dos resultados