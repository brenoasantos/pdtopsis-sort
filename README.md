# PDTOPSIS-Sort
Preference Disaggregation Technique for Order Preferences by Similarity to Ideal Solution (PDTOPSIS) - Sort, algorithm to support Multi-Criteria Decision Making (MCDM)

## Team
- baas@cin.ufpe.br
- grcc@cin.ufpe.br
- mmmj@cin.ufpe.br
- mldm@cin.ufpe.br

## Project Description
This project implements the PDTOPSIS-Sort algorithm, which is a Preference Disaggregation Technique for Order Preferences by Similarity to Ideal Solution (PDTOPSIS). It is designed to support Multi-Criteria Decision Making (MCDM).

### PDTOPSIS-Sort algorithm - Step by step:
1. Determine the Decision Matrix
2. Define the set of Reference Alternative
3. Determine the domain of each criterion
4. Infer the boundary profiles and the weights
5. Validate the parameters inferred in Step 4, before classifying the alternatives
6. Sorting process (more details in the paper)
7. Perform a sensitivity analysis over the results of the classification



## Installation
To run the application locally, follow these steps (bash):

1. Install streamlit:
	```pip3 install streamlit```

2. Create a virtual environment:
	```python3 -m venv streamlit_env```

3. Activate the virtual environment:
	```source streamlit_env/bin/activate```

4. Install the required libraries:
	```pip3 install -r requirements.txt```

## Link for the application in production:
[TOPSIS-Sort](https://pdtopsis-sort.onrender.com/)

## Usage
To run the interface, use the following command:

`streamlit run interface.py`

After running the interface, it is necessary to attach the `input.csv` and `matrixValues.csv` files.

## Limitations
This project has in issue in the step 4. The parameters are predetermined, as traditional TOPSIS method. In our program the number of classes is set to exactly 3, and they need to be named C1, C2 and C3.

## Suggestions
Use the CVXPY library to solve the problem in step 4. In the code itself, the library is already being imported, but it is commented.

## Improvement opportunities
Make the alternative classification phase dynamic. The way we did it, we treated the inputs with a single format: with three classes (C1, C2, C3).
