"""
seaborn package

https://seaborn.pydata.org/api.html
"""
#works with pandas dataframe only

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#dataset for plotting
from sklearn.datasets import load_boston
boston = load_boston()
df = pd.DataFrame(boston['data'], columns=boston['feature_names'])
df['price'] = boston['target']


#pairplot
###############################################################################
#https://seaborn.pydata.org/generated/seaborn.pairplot.html
if True:
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    sns.pairplot(df)
    plt.show()



#heatmap
###############################################################################
#https://seaborn.pydata.org/generated/seaborn.heatmap.html
if True:
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    sns.heatmap(df.corr())
    plt.show()
    
    #with annotation
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    sns.heatmap(df.corr(), annot=True)
    plt.show()
    
    #format annotation
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    sns.heatmap(df.corr(), annot=True, fmt=".1f")
    plt.show()
    
    #color scheam
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    sns.heatmap(df.corr(), annot=True, fmt=".1f", cmap="crest")
    plt.show()
    
    #range of the coloring
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    sns.heatmap(df.corr(), annot=True, fmt=".1f", cmap="crest", vmin=-0.6, vmax=0.6)
    plt.show()



#regplot
###############################################################################
#https://seaborn.pydata.org/generated/seaborn.regplot.html
if True:
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    sns.regplot(data=df,x='LSTAT',y='price')
    plt.show()
        