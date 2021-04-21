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


#Reference: https://seaborn.pydata.org/generated/seaborn.histplot.html
#Reference: https://www.python-graph-gallery.com/25-histogram-with-several-variables-seaborn
#Reference: https://stackoverflow.com/questions/42404154/increase-tick-label-font-size-in-seaborn

def plot_histograms(filename, plot_name, chart_title, x_series_one, x_series_two):  
    bin_number = 15
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle('{}: Histogram of {} variables (cm)'.format(plot_name,chart_title),fontsize = 25)

    sns.histplot(ax=axes[0, 0], data=iris_df, x = x_series_one, bins = bin_number, legend = False, kde = True,element = "step")
    sns.histplot(ax=axes[0, 1], data=iris_df, x = x_series_two, bins = bin_number, legend = False, kde = True,element ="step")
    sns.histplot(ax=axes[1, 0], data=iris_df, x = x_series_one, bins = bin_number, legend = False, hue = 'species',kde = True, element ="step") 
    hist_with_legend = sns.histplot(ax=axes[1, 1], data=iris_df, x = x_series_two, bins = bin_number, hue = 'species', kde = True, element = "step") 
    
    plt.setp(hist_with_legend.get_legend().get_texts(), fontsize='20') # for legend text
    plt.setp(hist_with_legend.get_legend().get_title(), fontsize='22') # for legend title
    
    for ax in plt.gcf().axes:
        x = ax.get_xlabel()
        y = ax.get_ylabel()
        ax.set_xlabel(x, fontsize=20)
        ax.set_ylabel(y, fontsize=0)

        plt.setp(ax.get_xticklabels(), fontsize=15)  
        plt.setp(ax.get_yticklabels(), fontsize=15)  
      
    fig.tight_layout() 
    plt.savefig('Images/' + filename +'.png')
 
plot_histograms('histograms_petals','Plot 1','Petals','petal_length','petal_width')
plot_histograms('histograms_sepals','Plot 2','Sepals','sepal_length','sepal_width')
