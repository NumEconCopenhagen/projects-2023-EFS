def prediction(row,rate):
    """Returns predicted values for each row variable, for the next year, given a growth rate"""
    return row[-1] * rate 

def plot_e(df, variable): 
    I = df['variables'] == variable
    ax=df.loc[I,:].plot(x='year', y='value', style='-o', legend=False)