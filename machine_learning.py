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


def knn_model_k_single(iris_data, iris_target_numeric):
#Reference: 
# Raschka, S, 2020, Example 2 - Visualizing the feature selection results, viewed 26 April 2021, 
# http://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/#example-2-visualizing-the-feature-selection-results

    # set model parameters
    knn = KNeighborsClassifier(n_neighbors=3)                   # call KNN model, will use 3 nearest neighbours
    efs1 = EFS(knn,                                             # specify model type for exhaustive feature selection library
            min_features=1,                                     # min features for comparison 
            max_features=4,                                     # min features for comparison 
            scoring='accuracy',                                 # accuracy method
            print_progress=True,             
            cv = 10)                                            # cross validation splits data into 5 groups of 30 observations 
    

    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']        # feature names into list
    efs1 = efs1.fit(iris_data, iris_target_numeric, custom_feature_names = feature_names) # fit data to efs1 pre-defined model
    df = pd.DataFrame.from_dict(efs1.get_metric_dict()).T                                 # uses get_metric_dict method on efs1 variable, efs1 is then converted from dict to dataframe
    df.sort_values('avg_score', inplace=True, ascending=False)                            # sort column in ascending order, inplace True means no copy is made
    print(df.head(10))

    # Create bar chart on mean performance of each subset of features
    fig, ax = plt.subplots(figsize=(12,9))                                                # create figure & axes
    y_pos = np.arange(len(df))                                                            # stores evenly spaced values that total the length of the df 

    plt.title('Plox X: KNeighborsClassifier Model Accuracy', fontsize = 20)               # add title
    ax.barh(y_pos, df['avg_score'], xerr=df['std_dev'])                                   # sets chart parameters
    ax.set_yticks(y_pos)                                                                  # sets y positions 
    ax.set_yticklabels(df['feature_names'])                                               # sets y labels
    ax.yaxis.tick_right()                                                                 # y axis ticks right
    ax.yaxis.set_label_position("right")                                                  # y axis labels right 
    ax.set_xlabel('Accuracy / 100')                                                       # sets x axis labels
    fig.tight_layout()                                                                    # adjust plot format to frame size
    plt.savefig('Images/' + 'knn_model_k_single' +'.png')   



def knn_model_k_multiple(iris_data, iris_target_numeric):
# Reference: 
# M, S, 2018, MachineLearning — KNN using scikit-learn, towards data science, viewed 27 April 2021, 
# https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75.


    # splits the numpy arrays into 20 : 80 sets for test size: train size  
    # random state control how splitting occuers, e.g a value of 0 will retain the same splitting method 
    X_train, x_test, Y_train, y_test = train_test_split(iris_data, iris_target_numeric, test_size=0.2, random_state = 0)

    k_range = range(1,26)        # define range for loop
    score = {}
    score_list = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k)                 # define knn as KNNClassifier and set k value as iterator          
        knn.fit(X_train,Y_train)                                    # Fit the k-nearest neighbors classifier from the training dataset
        y_pred = knn.predict(x_test)                                # Predict the class labels for the provided data
        score[k] = metrics.accuracy_score(y_test, y_pred)           # computes accuracy of test vs predicted datasets
        score_list.append(metrics.accuracy_score(y_test, y_pred))   # appends accuracy scores to score_list

    plt.clf()                                                                            # Clear figure
    plt.title('Plox X: KNeighborsClassifier Model % Accuracy', fontsize = 25)            # add title
    ax = sns.lineplot(x = k_range, y = score_list, color = 'blue')                       # seaborn line plot
    ax.set_xlabel('K value (Number of Neighbours)', fontsize=20)                         # set xlabel and fontsize
    ax.set_ylabel('% Accuracy / 100', fontsize=20)                                       # set ylabel and fontsize
    plt.setp(ax.get_xticklabels(), fontsize=15)                                          # set xticks and fontsize
    plt.setp(ax.get_yticklabels(), fontsize=15)                                          # set yticks and fontsize

    plt.tight_layout()                                                                   # adjust plot format to frame size
    plt.savefig('Images/' + 'knn_model_k_multiple' +'.png')   
    



if __name__ == '__main__':
    define_dataframes()
else:
    pass