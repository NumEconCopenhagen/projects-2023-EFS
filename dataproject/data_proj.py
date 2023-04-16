## goal: create a new column with values equal to the last one multiplied by 1.05

# function definition (can see more in lecture Data basics)
def prediction(row):
    return all.iloc[:,-1:] * 1.05 # should use row 

# This should be the call line the notebook
all['year2022'] = all.apply(prediction, axis=1)
all