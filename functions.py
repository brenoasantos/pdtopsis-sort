import numpy as np
import pandas as pd
import math

class PDTOPSIS_Sort:
    def __init__(self, matrix_values_filepath, input_filepath):
        self.matrix_values_csv = matrix_values_filepath
        self.input_csv = input_filepath
        self.decision_matrix = None
        self.reference_set = []
        self.domain = None
        self.weights = None
        self.profiles = None

        # carregar a matriz de decisão do arquivo CSV
        self.load_data()

    def load_data(self):
        # carrega os dados do arquivo CSV para a matriz de decisão
        # a matriz de decisão deve ser um DataFrame com alternativas nas linhas e critérios nas colunas
        self.decision_matrix = pd.read_csv(self.matrix_values_csv, delimiter=';', header=None)
        self.inputs = pd.read_csv(self.input_csv, delimiter=';')

        for index, row in self.inputs.iterrows():
            if row.ref_alternatives != " ":
                self.reference_set.append([row.ref_alternatives, row.ref_alt_class])
    
        
        print(self.reference_set)
    
    def determine_domain(self):
        # supondo que o domínio não é fornecido, calculamos com base na matriz de decisão.
        # caso contrário, esta função pode ser atualizada para aceitar domínios fornecidos.
        self.domain = {
            'ideal': self.decision_matrix.max(),
            'anti_ideal': self.decision_matrix.min()
        }

    def infer_parameters(self):
        # Placeholder para a inferência de parâmetros
        # Implementação real dependeria de técnicas de otimização
        self.weights = np.random.rand(self.decision_matrix.shape[1])  # Pesos aleatórios
        self.profiles = np.random.rand(self.decision_matrix.shape[1], 3)  # Três perfis fictícios

    def validate_parameters(self):
        # Imprimir pesos e perfis para validação manual
        print('Weights:', self.weights)
        print('Profiles:', self.profiles)

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
        self.distances_to_ideal = np.sqrt(np.sum((self.weighted_normalized_decision_matrix - self.v_star) ** 2, axis=1))
        self.distances_to_anti_ideal = np.sqrt(np.sum((self.weighted_normalized_decision_matrix - self.v_minus) ** 2, axis=1))

    def calculate_closeness_coefficients(self):
        '''
        Step 6.6: Calculate the closeness coefficients of each alternative and profile.
        '''
        self.closeness_coefficients = self.distances_to_anti_ideal / (self.distances_to_ideal + self.distances_to_anti_ideal)

    def classify_alternatives(self, profiles_closeness_coefficients):
        '''
        Step 6.7: Classify the alternatives by making comparisons between their closeness coefficients.
        '''
        self.classifications = np.digitize(self.closeness_coefficients, profiles_closeness_coefficients)
    
    def sensitivity_analysis(self):
        # Placeholder para análise de sensibilidade
        pass

    def run_algorithm(self):
        # Executar todos os passos do algoritmo sequencialmente
        self.determine_domain()
        self.infer_parameters()
        self.validate_parameters()
        self.classify_alternatives()
        self.sensitivity_analysis()
        # Output final: apresentação dos resultados