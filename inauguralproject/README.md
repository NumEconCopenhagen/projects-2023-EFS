# Inaugural project

The **results** of the project can be seen from running [Inaugural_Project_EFS.ipynb](Inaugural_Project_EFS.ipynb) whcih relys on [Household_Specialization_Model_EFS.py](Household_Specialization_Model_EFS.py).

**Dependencies:** 
Apart from a standard Anaconda Python 3 installation, the project requires no further packages.

All the answers to the assigment questions can be found by running sequentially the content of Inaugural_Project_EFS.ipynb. The code necessary to produce the necessary results has been implemented as an extension of the procided original class. The key modification are the introduction of missing functions and original functions:
- solve: solves the model for continuous values of HM HF LM LF;
- solve_wF_vec: to be used in both discrete and continuous cases to return a vector of optimal values of log HM/HF given different values of wF;
- estimate: estimates the optimal sigma an alpha to minimize deviation from empirical regression parameters;
- value_of_choiche: used to define the objective function for the solve function;
- modification: which estimates the optimal sigma, given an alpha, to minimize deviation from empirical regression parameters;
- tableHFHM: used to build a table returning values of HF/HM for every combination of sigma and alpha values.

At last some intuition has been provided on how to expand the model to minimize deviation from regression parameters while fixing alpha to 0.5.
