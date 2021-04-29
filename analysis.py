# This module has 1 Class & 5 functions [FN*..]:
# Author: Ciaran Moran

#[F1*]: read_iris_dataset()
        # reads in iris_data_set.txt & converts to dataframe

#[F2*]: *Summary* Class 
        # creates summary values for the .txt file

        #[F3*]: write_function()
        # instance method of Class that writes object to summary_file.txt

#[F4*]: plot_histograms_multi()
        # writes 8 histograms plots (2 figures) to Images Folder

#[F5*]: plot_boxplot()
        # writes 1 boxplot to Images Folder

#[F6*]: scatter_plot()
        # writes 6 scatter plots (1 figure) to Images Folder


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import patches as mpatches
import matplotlib.pyplot as plt
import os 

os.chdir(os.path.dirname(__file__))                                                              # change current directory to that of this module


#[F1*]    
def read_iris_dataset():
    iris_data_file = 'iris_data_set.txt'
    iris_df = pd.read_csv(iris_data_file, delimiter = ',', header = None)                        # read in Iris Dataset via Pandas Library
    iris_df.columns =['sepal_length', 'sepal_width', 'petal_length', 'petal_width','species']    # add columns headers
    return iris_df                                                                               # return dataframe
 
iris_df = read_iris_dataset()                                                                    # store dataframe in iris_df variable



#[F2*]
#Summary class which creates summary values for the .txt file and implements 
# a write function using the classes values
class Summary:  
    # constructor __init__ method:
        # called when object is Instantiated
        # Summary class initialisation
    def __init__(self, shape, data_types, null_count, species_count, desc_all_species, df_head, skewness_all_species, kurtosis_all_species, correlation):
        self.shape = shape
        self.data_types = data_types
        self.null_count = null_count
        self.species_count = species_count
        self.desc_all_species = desc_all_species
        self.df_head = df_head
        self.skewness_all_species = skewness_all_species
        self.kurtosis_all_species = kurtosis_all_species
        self.correlation = correlation

    #[F3*]
    # instance method
        # function in Summary class to write class variables
        # writes the summaries (w = overwrite) to text file, operation will make file if doesnt exist
    def write_function(self):
        with open('summary_file.txt', 'w') as f: 
            pd.set_option("display.precision", 2)
            spacing_variable = ('\n' * 3)
            f.write('Fisher Iris Dataset Summary' + spacing_variable)
            f.write('Number of columns = {} \nNumber of rows = {}{}'.format(self.shape[0],self.shape[1],spacing_variable)) 
            f.write('Column name:  Pandas dtypes: \n'+ str(self.data_types) + spacing_variable) 
            f.write('Column name:  Null Count: \n'+ str(self.null_count) + spacing_variable)
            f.write('Row Count per Species: \n'+ str(self.species_count) + spacing_variable)          
            f.write(' '*17 + 'First 5 rows of the Dataset\n' + str(self.df_head)  + spacing_variable)
            f.write(' '*17 + 'All Species : Summary Statistics \n' + str(self.desc_all_species) + spacing_variable)
            f.write('Distribution Skewness \n' + str(self.skewness_all_species) + spacing_variable)
            f.write('Distribution Kurtosis \n' + str(self.kurtosis_all_species) + spacing_variable)
            f.write(' '*17 + 'All Species : Correlation Statistics \n' + str(self.correlation) + spacing_variable)

# creates an instance of Summary class which passes in relevant variables and writes to .txt file
summary_data = Summary(
    (len(iris_df.axes[1]),len(iris_df.axes[0])),                                        # length/width of dataframe
    iris_df.dtypes,                                                                     # data-types of columns                                                           
    iris_df.isnull().sum(),                                                             # True for NaN / blank values
    iris_df.groupby('species').size(),                                                  # Computes group sizes                                 
    iris_df.describe(),                                                                 # dataframe summary 
    iris_df.head(5),                                                                    # first 5 rows of dataframe
    iris_df.skew(),                                                                     # returns skew of all iris features                        
    iris_df.kurtosis(),                                                                 # returns kurtosis of all iris features
    iris_df.corr(),                                                                     # Compute pairwise Pearsons correlation of columns
)
summary_data.write_function() 



#[F4*]    
#Reference used to generate plot: 
# [1] Holtz, Y, 2021, Histogram with several variables with Seaborn, viewed 20 April 2021, 
#     https://www.python-graph-gallery.com/25-histogram-with-several-variables-seaborn
# [2] Waskom, M, 2021, seaborn.histplot, viewed 20 April 2021, https://seaborn.pydata.org/generated/seaborn.histplot.html.
def plot_histograms_multi(filename, plot_name, chart_title, x_series_one, x_series_two):  
    
    plt.clf()  # clear current figure 

    # define fig and axex as plt.subplot() function
    # plt.subplots returns a figure and set of subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
 
    # generate 4 plots in the defined subplots locations
    bin_number = 15
    sns.histplot(ax=axes[0, 0], data = iris_df, x = x_series_one, bins = bin_number, legend = False, kde = True, element = "step")
    sns.histplot(ax=axes[0, 1], data = iris_df, x = x_series_two, bins = bin_number, legend = False, kde = True, element ="step")
    sns.histplot(ax=axes[1, 0], data = iris_df, x = x_series_one, bins = bin_number, legend = False, hue = 'species', kde = True, element ="step") 
    plot_four = sns.histplot(ax=axes[1, 1], data=iris_df, x = x_series_two, bins = bin_number, hue = 'species', kde = True, element = "step") 
    
    # set figure title and call get_legend() on plot_four
    fig.suptitle('{}: Histogram of {} variables (cm)'.format(plot_name,chart_title),fontsize = 25) 
    plt.setp(plot_four.get_legend().get_texts(), fontsize='20') # plot four: return get_lenged() into artist object & call get_texts() to change text fontsize
    plt.setp(plot_four.get_legend().get_title(), fontsize='22') # plot four: return get_lenged() into artist object & call get_title() to change title fontsize
   
    # reformat plot
    for ax in plt.gcf().axes:                                   # for subplots in figure
        x = ax.get_xlabel()                                     # define x_label as x
        y = ax.get_ylabel()                                     # define y_label as y
        ax.set_xlabel(x, fontsize=20)                           # set xlabel to x variable and alter fontsize
        ax.set_ylabel(y, fontsize=0)                            # set xlabel to y variable and hide labels            
                                                                
        plt.setp(ax.get_xticklabels(), fontsize=15)             # return xticklabels() into artist object and change fontsize
        plt.setp(ax.get_yticklabels(), fontsize=15)             # return yticklabels() into artist object and change fontsize           
      
    fig.tight_layout()                                          # optimise fit of subplots to figure
    plt.savefig('Images/' + filename +'.png')
 


#[F5*]  
#Reference used to generate plot: 
# [1] C, J, 2020, Create a single legend for multiple plot in matplotlib, seaborn, stack overflow, viewed 21 April 2021, 
#     https://stackoverflow.com/questions/62252493/create-a-single-legend-for-multiple-plot-in-matplotlib-seaborn.
# [2] Waskom, M, 2021, seaborn.boxplot, viewed 21 April 2021, https://seaborn.pydata.org/generated/seaborn.boxplot.html.
def plot_boxplot():
    plt.clf() 
    fig, axes = plt.subplots(1,4, figsize=(20, 12)) 

    sns.boxplot(ax=axes[0], x = iris_df["species"], y = iris_df["petal_length"], data = iris_df, width=0.75)
    sns.boxplot(ax=axes[1], x=iris_df["species"], y=iris_df["petal_width"], data = iris_df, width=0.75)
    sns.boxplot(ax=axes[2], x=iris_df["species"], y=iris_df["sepal_length"], data = iris_df, width=0.75)
    sns.boxplot(ax=axes[3], x=iris_df["species"], y=iris_df["sepal_width"], data  =iris_df, width=0.75)


    # define a list of species names  
    # call Patch method frim mpathces lib. and set colour                            
    legend_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']      
    setosa = mpatches.Patch(color='steelblue')                               
    versi = mpatches.Patch(color='darkorange')
    virgi = mpatches.Patch(color='green')

    # plot legend and use lenged_label list as handles
    plt.legend(title = False, labels=legend_labels,                        
              handles=[setosa, versi, virgi], bbox_to_anchor=(-0.05, 1.09),
              fancybox=False, shadow=False, ncol=3, loc='upper right', fontsize = 25)
    
    fig.suptitle('Plot 3 : Boxplot of Iris dependant variables (cm)', fontsize = 30) 
    
    for ax in plt.gcf().axes:  
        x = ax.get_xlabel()
        y = ax.get_ylabel()
        ax.set_xlabel(x, fontsize=26)
        ax.set_ylabel(y, fontsize=26)
        ax.set_ylim([0, 8])                                                  

        ax.set_xticks([])
        plt.setp(ax.get_xticklabels(), fontsize=22)  
        plt.setp(ax.get_yticklabels(), fontsize=17.5) 
    
    plt.savefig('Images/' + 'plot3_box_plots' +'.png')



#[F6*] 
#Reference used to generate plot: 
# Waskom, M, 2021, seaborn.seaborn.regplot, viewed 23 April 2021, https://seaborn.pydata.org/generated/seaborn.regplot.html.
def scatter_plot():  
    plt.clf()    
    fig, axes = plt.subplots(2, 3, figsize=(22, 18))
    plt.subplots_adjust(wspace=0.2,hspace=0.4)
    
    sns.regplot(ax=axes[0, 0], data=iris_df, x='petal_length', y='petal_width')
    sns.regplot(ax=axes[0, 1], data=iris_df, x='petal_length', y='sepal_length')
    sns.regplot(ax=axes[0, 2], data=iris_df, x='petal_length', y='sepal_width') 
    sns.regplot(ax=axes[1, 0], data=iris_df, x='sepal_length', y='sepal_width')
    sns.regplot(ax=axes[1, 1], data=iris_df, x='sepal_length', y='petal_width') 
    sns.regplot(ax=axes[1, 2], data=iris_df, x='sepal_width',  y='petal_width')

    fig.suptitle('Plot 4: Scatter Plot of all variables (units = cm)',fontsize = 25)

    for ax in plt.gcf().axes:
        x = ax.get_xlabel()
        y = ax.get_ylabel()
        ax.set_xlabel(x, fontsize=20)
        ax.set_ylabel(y, fontsize=20)
        ax.set_xlim([0, 8])
        ax.set_ylim([0, 8])
        plt.setp(ax.get_xticklabels(), fontsize=15)  
        plt.setp(ax.get_yticklabels(), fontsize=15)  
    
    fig.tight_layout()
    # adjust subplots spacing  
    plt.subplots_adjust(wspace=0.25, hspace = 0.25, top = 0.95)                 
    plt.savefig('Images/' + 'plot4_scatter_plots' +'.png')   


    
plot_histograms_multi('plot1_histograms_petals','Plot 1','Petals','petal_length','petal_width')
plot_histograms_multi('plot2_histograms_sepals','Plot 2','Sepals','sepal_length','sepal_width')
plot_boxplot()
scatter_plot()



