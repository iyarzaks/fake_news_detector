import pandas as pd
from project_utils import *
from sklearn import linear_model, datasets


data = pd.read_csv('Classified_Data_kaggle.csv', engine='python')
artcles_list = data['Body'].values.astype('U').tolist()
labels = data['Label (1=true)'].values.astype('U').tolist()
words = read_json("top_50.json")
#to_bag_of_words(artcles_list,labels)

df_with_imp_words = build_table_form_words(data,words)
X = df_with_imp_words.iloc[:100,:50]
print (X)
Y = df_with_imp_words['Label (1=true)']
Y = Y.iloc[:100]
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X, Y)
print (logreg.coef_)