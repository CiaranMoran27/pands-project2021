# This module uses Exhaustive Feature Selector (EHS) wrapper method to evaluate 
# mean accuracy % of Knearest Neighbouts Machine Learning Model and 
# also evaluates a range of k values to obtain info. on more effective K values.
# Author Ciaran Moran


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analysis import read_iris_dataset

# machine learning libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS



def define_dataframes():
    iris_df = read_iris_dataset()                                                                    # store iris dataframe (function was imported)
    iris_data = iris_df[["petal_length", "petal_width", "sepal_length","sepal_width"]].to_numpy()    # seperate feature columns into numpy list of lists
    iris_target = iris_df[iris_df.columns[4]]                                                        # store iris Species series in variable
    label_encoder = preprocessing.LabelEncoder()                                                     # instansiate label encoder object
    iris_target_numeric = label_encoder.fit_transform(iris_target)                                   # fit species series into label_encoder object (as numpy array) using fit_transform function-> converts species to numbers
    
    # pass relevant numpy arrays
    knn_model_k_single(iris_data, iris_target_numeric)                                               
    knn_model_k_multiple(iris_data, iris_target_numeric)                                             
