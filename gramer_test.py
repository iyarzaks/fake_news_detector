from project_utils import *
import sys
import pandas as pd
all_data = pd.read_csv('newOrderAlonParams.csv', engine='python')
data = all_data.loc[all_data['Label (1=true)'].isin(['0','1'])]
print (data.shape)
X = data.iloc[:,[4,5,6,7,8,9,10,11,12,13,15,17,19,21]]
#print (X)
Y = data['Label (1=true)']
#print (Y)
clf = lr_func(X, Y)
coef_plot(clf,X)
