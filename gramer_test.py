from project_utils import *
import sys
import pandas as pd
import difflib
all_data = pd.read_csv('initDataPerArticle.csv', engine='python')
all_data.apply(str)
orginal_data = pd.read_csv('Classified_Data_kaggle.csv', engine='python')
print (all_data.shape)
all_data['label'] = ""
for index,row in all_data.iterrows():
    all_data.loc['label'].iloc[index] = orginal_data.loc[orginal_data['URLs'] == row.loc['URL'].replace('\n','',1)]
print (all_data.shape)
data = all_data.loc[all_data['label'].isin(['0','1'])]
X = all_data.iloc[:,1:10]
Y = data['label']
# Y_train = Y.iloc[:4000]
clf = svm_func(X, Y)
# coef_plot(clf,df_with_imp_words)
