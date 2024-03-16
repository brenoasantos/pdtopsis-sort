from functions import PDTOPSIS_Sort

# essa main será usada quando formos testar a interface
'''
def main():
    pdtopsis = PDTOPSIS_Sort(matrix_values_filepath="app_input/matrixValues.csv", input_filepath="app_input/input.csv")
    pdtopsis.run_algorithm()
'''
def main():
    pdtopsis = PDTOPSIS_Sort(matrix_values_filepath="matrixValues.csv", input_filepath="input.csv")
    pdtopsis.run_algorithm()

if __name__ == "__main__":
    main()