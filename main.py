import pandas as pd
from functions import PDTOPSIS_Sort

def main():
    pdtopsis = PDTOPSIS_Sort(matrix_values_filepath="app_input/matrixValues.csv", input_filepath="app_input/input.csv")
    pdtopsis.run_algorithm()