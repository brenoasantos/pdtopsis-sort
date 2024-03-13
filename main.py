import pandas as pd
from functions import PDTOPSIS_Sort

# # leitura do csv que contém as entradas do programa
# input = pd.read_csv("input.csv", delimiter=';')

# #leitura do csv que contém os valores da matriz de decisão
# decisionMatrix = pd.read_csv("matrixValues.csv", delimiter=';', header=None)

# criterion = list(input.criterion)
# alternatives = list(input.alternatives)

# x = criterion.index(' ')
# del criterion[x:]

# print(decisionMatrix)

def main():
    # Substitua 'path_to_csv_file.csv' pelo caminho do seu arquivo CSV
    pdtopsis = PDTOPSIS_Sort(matrix_values_filepath="matrixValues.csv", input_filepath="input.csv")
    pdtopsis.run_algorithm()

if __name__ == "__main__":
    main()