import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os 

os.chdir(os.path.dirname(__file__))                                                                           # change current directory to that of this module


def read_iris_dataset():
    iris_data_file = 'iris_data_set.txt'
    iris_dataframe = pd.read_csv(iris_data_file, delimiter = ',', header = None)                              # read in Iris Dataset via Pandas Library
    iris_dataframe.columns =['sepal_length', 'sepal_width', 'petal_length', 'petal_width','species']          # add columns headers
    return iris_dataframe
    
iris_dataframe = read_iris_dataset()        


def summary_variables():
    
    shape = (len(iris_dataframe.axes[1]),len(iris_dataframe.axes[0]))
    data_types = iris_dataframe.dtypes                                                            
    species_count = iris_dataframe.groupby('species').size()
    null_count = iris_dataframe.isnull().sum()   # True for NaN / blank values
    desc_all_species = iris_dataframe.describe()
    correlation = iris_dataframe.corr()
    pd.set_option("display.precision", 2)

    # Check if numeric columns can be converted to integer
    numeric_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    try:
        iris_dataframe[numeric_columns].astype(float).astype(int)
        integer_check = 'The following columns only contain numeric values:\n{}.'.format(numeric_columns)      
    except ValueError:
        integer_check = 'Not all entires in the following columns contain numeric values \n{}.'.format(numeric_columns)  
    
    return shape, data_types, null_count,species_count, integer_check, desc_all_species, correlation
    
summary_tuple = summary_variables()   


def write_summary_variables(summary_tuple):
    with open('summary_file.txt', 'w') as f:
        
        skip_three_lines = ('\n' * 3)
        f.write('Fisher Iris Dataset Summary' + skip_three_lines)
        f.write('Number of columns = {} \nNumber of rows = {}{}'.format(summary_tuple[0][0],summary_tuple[0][1],skip_three_lines)) 
        f.write('Column name:  Data Types: \n'+ str(summary_tuple[1]) + skip_three_lines) 
        f.write('Column name:  Null Count: : \n'+ str(summary_tuple[2]) + skip_three_lines)
        f.write('Row Count per Species: \n'+ str(summary_tuple[3]) + skip_three_lines)  
        f.write(summary_tuple[4] + '\n' + skip_three_lines)
        f.write(' '*17 + 'All Species : Summary Statistics \n' + str(summary_tuple[5]) + skip_three_lines)
        f.write(' '*17 + 'All Species : Correlation Statistics \n' + str(summary_tuple[6]) + skip_three_lines)

write_summary_variables(summary_tuple)