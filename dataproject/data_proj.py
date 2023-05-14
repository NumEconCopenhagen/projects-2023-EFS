import pandas as pd
import ipywidgets as widgets


def prediction(row,rate):
    """Returns predicted values for each row variable, for the next year, given a growth rate"""

    return row[-1] * rate 

def handle_gdp_data(gdp_dst):
    """ Download the dataset and prepare it by selecting the desired variables, dropping 
        unimportant ones, renaming for the sake of clarity and indexing, and transposing it"""

    #Define parameters dictionary to select only specified values (rows) of dataset:
    par_gdp = {'table': 'nrhp',
    'format': 'BULK',
    'lang': 'en',
    'variables': [{'code': 'OMRÅDE', 'values': ['000']},
    {'code': 'TRANSAKT', 'values': ['B1GQD']},
    {'code': 'PRISENHED', 'values': ['V_C']},
    {'code': 'Tid', 'values': ['>1993<=2021']}]}

    #Download the specific dataset by specified parameters:
    gdp = gdp_dst.get_data(params=par_gdp)

    #Rename columns:
    gdp.rename(columns = {'OMRÅDE':'Area', 
                        'PRISENHED':'Price unit', 
                        'TID':'variables', #helpfull later
                        'INDHOLD':'GDP'}, inplace=True)

    #Drop unimportant variables
    gdp.drop(['TRANSAKT', 'Area', 'Price unit'], axis='columns', inplace=True)

    #Make column names a mix of text and numbers (without spaces) and set index:
    import string 

    for value in gdp['variables'].values:
        gdp.loc[gdp['variables'].values == value,['variables']] = 'value'+str(value)
    gdp = gdp.set_index('variables')

    #Transpose
    gdp = gdp.T

    return gdp

def handle_consumption_data(cop):
    """ Prepare the dataset by droping unecessary information, renaming variables to 
        make it easier to work with them and resetting the index"""
    
    #Drop NaN columns:
    drop_these = ['Unnamed: ' + str(num) for num in range(2)] # use list comprehension to create list of columns
    cop.drop(drop_these, axis=1, inplace=True) # axis = 1 -> columns, inplace=True -> changed, no copy made

    #Rename consumption and year columns:
    cop.rename(columns = {'Unnamed: 2':'variables'}, inplace=True)
    col_dict = {}
    col_dict = {str(i) : f'value{i}' for i in range(1994,2021+1)}
    cop.rename(columns = col_dict, inplace=True)

    #Drop unimportant variables:
    I = cop.variables.str.contains('Household textiles')
    cop.loc[I, :]
    cop = cop.loc[I == False] # keeping everything else

    #Reset the index
    cop.reset_index(inplace = True, drop = True) # Drop old index too
    cop.iloc[0:7,:]

    #Remove numbers from consumption categories:
    for value in cop['variables'].values:
        cop.loc[cop['variables'].values == value,['variables']] = value.strip('0123456789.')

    cop.loc[0,['variables']] = 'Total consumption'

    #Set variables as index:
    cop = cop.set_index('variables')
    
    return cop

def concatenate_datasets(cop, gdp):
    """ Check if the datasets have the same variables and concatenate them"""

    #Check if they have the same variables 
    different_years = [y for y in cop.columns.unique() if y not in gdp.columns.unique()] 
    print(f'Columns (years) found in cop data but not in gdp: {different_years}')

    if different_years != []:
        return print("Not all variables are present in both datasets")
    else:
       #Concatenate them
        all = pd.concat([cop,gdp])
        return all

   

def accomodate_data(all, scalar = 1000):
    """ Standardize all units """

    #Rename index:
    all.index.names = ['variables']

    #Consumption is in DKK while GDP (per capita) is in 1000 DKK. It will be homogenized towards the unitary value.
    for i in all.index.values:
        if i != 'GDP':
            all[all.index == i] = all[all.index == i] / scalar
    
    return all

def analysis(all):
    """ Make a prediction for 2022 and check each variable over GDP """

    #Create new column, year2022, which contains values 
    #given a 0.05 growth rate prediction of every variable in year 2022:
    all['value2022'] = all.apply(prediction, rate=1.05, axis=1)

    #Check consumption of each variable over GDP:
    for val in all.index:
        all.loc[val + "/GDP"] = all.loc[val] / all.loc["GDP"]

    #Set decimal units
    all = all.astype(float).round(decimals=2)
    
    return all

def handle_data_graph(all):
    """ Change the data format to make it suitable for graphing"""

    #Reset the index:
    all = all.reset_index()

    #Transform dataframe from wide to long format:
    all_long = pd.wide_to_long(all, stubnames='value', i='variables', j='year')

    #Save a copy of the final format of our dataset (uncomment to run the code):
    #all_long.to_csv('data/FU07_cp_long.xlsx', index=False)

    #Reset the index again
    all_long = all_long.reset_index()

    return all_long

def plot_graph(all_long):
    """ Plot an interactive graph"""

    def plot_e(df, variable): 
        I = df['variables'] == variable
        ax=df.loc[I,:].plot(x='year', y='value', style='-o', legend=False)

    return widgets.interact(plot_e, 
            df = widgets.fixed(all_long),
            variable = widgets.Dropdown(description='variables', 
                                            options=all_long.variables.unique(), 
                                            value='Total consumption')
        ); 

