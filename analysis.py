import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os 

os.chdir(os.path.dirname(__file__))                                                                           # change current directory to that of this module


def read_iris_dataset():
    iris_data_file = 'Iris_Data_Set.txt'
    iris_dataframe = pd.read_csv(iris_data_file, delimiter = ',', header = None)                              # read in Iris Dataset via Pandas Library
    iris_dataframe.columns =['sepal_length', 'sepal_width', 'petal_length', 'petal_width','species']          # add columns headers
    return iris_dataframe
    
iris_dataframe = read_iris_dataset()        