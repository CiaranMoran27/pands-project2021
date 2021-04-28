import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import patches as mpatches
import matplotlib.pyplot as plt
import os 

os.chdir(os.path.dirname(__file__))                                                              # change current directory to that of this module


def read_iris_dataset():
    iris_data_file = 'iris_data_set.txt'
    iris_df = pd.read_csv(iris_data_file, delimiter = ',', header = None)                        # read in Iris Dataset via Pandas Library
    iris_df.columns =['sepal_length', 'sepal_width', 'petal_length', 'petal_width','species']    # add columns headers
    return iris_df                                                                               # return dataframe
 
iris_df = read_iris_dataset()                                                                    # store dataframe in iris_df variable




class summary_creator:  
    # constructor __init__ method:
        # called when object is Instantiated
        # allows summary_creator class to initialise its attributes

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
    

    # instance method
        # calls object attributes relevant to summary
        # writes the summaries (w = overwrite) to text file, operation will make file if doesnt exist
    def write_function(self):
        with open('summary_file.txt', 'w') as f: 

            spacing_variable = ('\n' * 3)
            f.write('Fisher Iris Dataset Summary' + spacing_variable)
            f.write('Number of columns = {} \nNumber of rows = {}{}'.format(summary_creator_val.shape[0],summary_creator_val.shape[1],spacing_variable)) 
            f.write('Column name:  Pandas dtypes: \n'+ str(summary_creator_val.data_types) + spacing_variable) 
            f.write('Column name:  Null Count: \n'+ str(summary_creator_val.null_count) + spacing_variable)
            f.write('Row Count per Species: \n'+ str(summary_creator_val.species_count) + spacing_variable)          
            f.write(' '*17 + 'First 5 rows of the Dataset\n' + str(summary_creator_val.df_head)  + spacing_variable)
            f.write(' '*17 + 'All Species : Summary Statistics \n' + str(summary_creator_val.desc_all_species) + spacing_variable)
            f.write('Distribution Skewness \n' + str(summary_creator_val.skewness_all_species) + spacing_variable)
            f.write('Distribution Kurtosis \n' + str(summary_creator_val.kurtosis_all_species) + spacing_variable)
            f.write(' '*17 + 'All Species : Correlation Statistics \n' + str(summary_creator_val.correlation) + spacing_variable)


def summary_variables():                                                                     # setting summary variables via pandas methods
        shape = (len(iris_df.axes[1]),len(iris_df.axes[0]))                                  # length/width of dataframe
        data_types = iris_df.dtypes                                                          # data-types of columns                                                           
        null_count = iris_df.isnull().sum()                                                  # True for NaN / blank values
        species_count = iris_df.groupby('species').size()                                    # Computes group sizes                                 
        desc_all_species = iris_df.describe()                                                # dataframe summary 
        df_head = iris_df.head(5)                                                            # first 5 rows of dataframe
        skewness = iris_df.skew()                                                            # returns skew of all iris features                        
        kurtosis = iris_df.kurtosis()                                                        # returns kurtosis of all iris features
        correlation = iris_df.corr()                                                         # Compute pairwise Pearsons correlation of columns
        pd.set_option("display.precision", 2)                                                # display data to 2 decimal places

        # Instantiate an object of type summary_creator & return object
        summary_creator_val = summary_creator(shape, data_types, null_count, species_count, 
        desc_all_species, df_head, skewness, kurtosis, correlation)
        return summary_creator_val

summary_creator_val = summary_variables()                                               # set variable to object (via calling function and returning object)
summary_creator_val.write_function()                                                    # call write_function() instance method of summary_creator class



def plot_histograms(filename, plot_name, chart_title, x_series_one, x_series_two):  
#Reference: 
# [1] Holtz, Y, 2021, Histogram with several variables with Seaborn, viewed 20 April 2021, 
#     https://www.python-graph-gallery.com/25-histogram-with-several-variables-seaborn
# [2] Waskom, M, 2021, seaborn.histplot, viewed 20 April 2021, https://seaborn.pydata.org/generated/seaborn.histplot.html.
    bin_number = 15

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle('{}: Histogram of {} variables (cm)'.format(plot_name,chart_title),fontsize = 25)

    sns.histplot(ax=axes[0, 0], data = iris_df, x = x_series_one, bins = bin_number, legend = False, kde = True, element = "step")
    sns.histplot(ax=axes[0, 1], data = iris_df, x = x_series_two, bins = bin_number, legend = False, kde = True, element ="step")
    sns.histplot(ax=axes[1, 0], data = iris_df, x = x_series_one, bins = bin_number, legend = False, hue = 'species', kde = True, element ="step") 
    plot_four = sns.histplot(ax=axes[1, 1], data=iris_df, x = x_series_two, bins = bin_number, hue = 'species', kde = True, element = "step") 
    
    plt.setp(plot_four.get_legend().get_texts(), fontsize='20') # for legend text
    plt.setp(plot_four.get_legend().get_title(), fontsize='22') # for legend title
    
    for ax in plt.gcf().axes:
        x = ax.get_xlabel()
        y = ax.get_ylabel()
        ax.set_xlabel(x, fontsize=20)
        ax.set_ylabel(y, fontsize=0)

        plt.setp(ax.get_xticklabels(), fontsize=15)  
        plt.setp(ax.get_yticklabels(), fontsize=15)  
      
    fig.tight_layout() 
    plt.savefig('Images/' + filename +'.png')
 


def plot_boxplot():
# [1] C, J, 2020, Create a single legend for multiple plot in matplotlib, seaborn, stack overflow, viewed 21 April 2021, 
#     https://stackoverflow.com/questions/62252493/create-a-single-legend-for-multiple-plot-in-matplotlib-seaborn.
# [2] Waskom, M, 2021, seaborn.boxplot, viewed 21 April 2021, https://seaborn.pydata.org/generated/seaborn.boxplot.html.

    fig, axes = plt.subplots(1,4, figsize=(26, 14))
    fig.suptitle('Fig 3 : Boxplot of Iris dependant variables (cm)', fontsize = 30)

    sns.boxplot(ax=axes[0], x = iris_df["species"], y = iris_df["petal_length"], data = iris_df, width=0.75)
    sns.boxplot(ax=axes[1], x=iris_df["species"], y=iris_df["petal_width"], data = iris_df, width=0.75)
    sns.boxplot(ax=axes[2], x=iris_df["species"], y=iris_df["sepal_length"], data = iris_df, width=0.75)
    sns.boxplot(ax=axes[3], x=iris_df["species"], y=iris_df["sepal_width"], data  =iris_df, width=0.75)

    legend_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    setosa = mpatches.Patch(color='steelblue')
    versi = mpatches.Patch(color='darkorange')
    virgi = mpatches.Patch(color='green')

    plt.legend(title = False, labels=legend_labels,
              handles=[setosa, versi, virgi], bbox_to_anchor=(-0.05, 1.09),
              fancybox=False, shadow=False, ncol=3, loc='upper right', fontsize = 25)

    for ax in plt.gcf().axes:  
        x = ax.get_xlabel()
        y = ax.get_ylabel()
        ax.set_xlabel(x, fontsize=22)
        ax.set_ylabel(y, fontsize=22)
        ax.set_ylim([0, 8])

        ax.set_xticks([])
        plt.setp(ax.get_xticklabels(), fontsize=20)  
        plt.setp(ax.get_yticklabels(), fontsize=15) 

    plt.savefig('Images/' + 'plot3_box_plots' +'.png')



def scatter_plot():     
# Reference:
# Waskom, M, 2021, seaborn.seaborn.regplot, viewed 23 April 2021, https://seaborn.pydata.org/generated/seaborn.regplot.html.
    fig, axes = plt.subplots(2, 3, figsize=(22, 18))
    plt.subplots_adjust(wspace=0.2,hspace=0.4)
    fig.suptitle('Plot 4: Scatter Plot of all variables (units = cm)',fontsize = 25)

    sns.regplot(ax=axes[0, 0], data=iris_df, x='petal_length', y='petal_width')
    sns.regplot(ax=axes[0, 1], data=iris_df, x='petal_length', y='sepal_length')
    sns.regplot(ax=axes[0, 2], data=iris_df, x='petal_length', y='sepal_width') 
    sns.regplot(ax=axes[1, 0], data=iris_df, x='sepal_length', y='sepal_width')
    sns.regplot(ax=axes[1, 1], data=iris_df, x='sepal_length', y='petal_width') 
    sns.regplot(ax=axes[1, 2], data=iris_df, x='sepal_width',  y='petal_width')

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
    plt.subplots_adjust(wspace=0.25, hspace = 0.25, top = 0.95)     
    plt.savefig('Images/' + 'plot4_scatter_plots' +'.png')   


    
    
if __name__ == '__main__':
    plot_histograms('plot1_histograms_petals','Plot 1','Petals','petal_length','petal_width')
    plot_histograms('plot2_histograms_sepals','Plot 2','Sepals','sepal_length','sepal_width')
    plot_boxplot()
    scatter_plot()

else:
    pass
