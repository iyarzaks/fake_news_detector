from project_utils import *
from words_test import *
import pandas as pd
all_data = pd.read_csv('initDataPerArticle.csv', engine='python')
data = all_data.loc[all_data['Label (1=true)'].isin(['0','1'])]
X = all_data.iloc[:4000,1:10]
Y = data['Label (1=true)']
Y_train = Y.iloc[:4000]
clf = svm_func(X, Y_train)
coef_plot(clf,df_with_imp_words)
