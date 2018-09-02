import pandas as pd
from project_utils import *
from sklearn import linear_model, datasets
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegressionCV

data = pd.read_csv('Classified_Data_kaggle.csv', engine='python')
artcles_list = data['Body'].values.astype('U').tolist()
labels = data['Label (1=true)'].values.astype('U').tolist()
words = read_json("top_50.json")
#to_bag_of_words(artcles_list,labels)

df_with_imp_words = build_table_form_words(data,words)
X = df_with_imp_words.iloc[:300,:50]
Y = df_with_imp_words['Label (1=true)']
Y = Y.iloc[:300]
# logreg = linear_model.LogisticRegression(C=1e5)
# logreg.fit(X, Y)
# print (np.around(logreg.coef_),2)
# X = df_with_imp_words.iloc[100:200,:50]
# res = logreg.predict(X)

lr = LogisticRegressionCV(Cs=15)
lr = lr.fit(X, Y)
print (lr.scores_)
coef_list = lr.coef_[0].tolist()
print ([format(a, '.2f') for a in coef_list])