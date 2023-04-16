def prediction(row,rate):
    """Returns predicted values for each row variable, for the next year, given a growth rate"""
    return row[-1] * rate 