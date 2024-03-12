import pandas as pd

# leitura do csv que contém as entradas do programa
input = pd.read_csv("input.csv", delimiter=';')

#leitura do csv que contém os valores da matriz de decisão
decisionMatrix = pd.read_csv("matrixValues.csv", delimiter=';', header=None)

criterion = list(input.criterion)
alternatives = list(input.alternatives)

x = criterion.index(' ')
del criterion[x:]

print(decisionMatrix)