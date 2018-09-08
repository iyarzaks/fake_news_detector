from project_utils import *
import sys
import pandas as pd
all_data = pd.read_csv('initDataPerArticle.csv', engine='python')
print (all_data.shape)
data = all_data.loc[all_data['label'].isin(['0','1'])]
print (data.shape)
X = data.iloc[:,1:10]
#print (X)
Y = data['label']
#print (Y)
clf = lr_func(X, Y)
coef_plot(clf,X)
