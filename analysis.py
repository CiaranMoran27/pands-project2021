import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os 

os.chdir(os.path.dirname(__file__))                                                                    # change current directory to that of this module


def read_iris_dataset():
    iris_data_file = 'iris_data_set.txt'
    iris_df = pd.read_csv(iris_data_file, delimiter = ',', header = None)                              # read in Iris Dataset via Pandas Library
    iris_df.columns =['sepal_length', 'sepal_width', 'petal_length', 'petal_width','species']          # add columns headers
    return iris_df

iris_df = read_iris_dataset()

    

def summary_variables():
    
    shape = (len(iris_df.axes[1]),len(iris_df.axes[0]))                                  # count # rows on each axis of array
    data_types = iris_df.dtypes                                                            
    null_count = iris_df.isnull().sum()                                                  # True for NaN / blank values
    species_count = iris_df.groupby('species').size()                                    
    desc_all_species = iris_df.describe()
    df_head = iris_df.head(5)
    skewness_all_species = iris_df.skew()
    kurtosis_all_species = iris_df.kurtosis()
    correlation = iris_df.corr()
    pd.set_option("display.precision", 2)
    
    return shape, data_types, null_count, species_count, desc_all_species, df_head, skewness_all_species, kurtosis_all_species, correlation
    
summary_tuple = summary_variables()   



def write_summary_variables(summary_tuple):
    with open('summary_file.txt', 'w') as f:

        skip_three_lines = ('\n' * 3)
        
        f.write('Fisher Iris Dataset Summary' + skip_three_lines)
        f.write('Number of columns = {} \nNumber of rows = {}{}'.format(summary_tuple[0][0],summary_tuple[0][1],skip_three_lines)) 
        f.write('Column name:  Pandas dtypes: \n'+ str(summary_tuple[1]) + skip_three_lines) 
        f.write('Column name:  Null Count: \n'+ str(summary_tuple[2]) + skip_three_lines)
        f.write('Row Count per Species: \n'+ str(summary_tuple[3]) + skip_three_lines)  
        f.write(' '*17 + 'First 5 rows of the Dataset\n' + str(summary_tuple[4])  + skip_three_lines)
        f.write(' '*17 + 'All Species : Summary Statistics \n' + str(summary_tuple[5]) + skip_three_lines)
        f.write('Distribution Skewness \n' + str(summary_tuple[6]) + skip_three_lines)
        f.write('Distribution Kurtosis \n' + str(summary_tuple[7]) + skip_three_lines)
        f.write(' '*17 + 'All Species : Correlation Statistics \n' + str(summary_tuple[8]) + skip_three_lines)

write_summary_variables(summary_tuple)