import pandas as pd
from project_utils import *
from sklearn import linear_model, datasets
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegressionCV
import sys
#from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


all_data = pd.read_csv('Classified_Data_kaggle.csv', engine='python')
print (all_data.shape)
bad_input = all_data.loc[~all_data['Label (1=true)'].isin(['0','1'])]
print_to_json(bad_input.index.values.tolist(),"bad_input_indexes")
data = all_data.loc[all_data['Label (1=true)'].isin(['0','1'])]
print (data.shape)

articles_list = data['Body'].iloc[:500].values.astype('U').tolist()
labels = data['Label (1=true)'].iloc[:500].values.astype('U').tolist()
to_bag_of_words(articles_list, labels, "top_50.json.1","words_importance.json.1",200)
words = read_json("top_50.json.1")
#to_bag_of_words(artcles_list,labels)

df_with_imp_words = build_table_form_words(data,words)
X = df_with_imp_words.iloc[500:3000,:200]
Y = data['Label (1=true)']
Y_train = Y.iloc[500:3000]

#fold = KFold(len(Y_train), n_folds=10, shuffle=True, random_state=777)
#lr = lr_func(X,Y_train)


#mean_scores = np.mean(scores, axis=0)
# coef_plot(lr,df_with_imp_words)

#manual_check(df_with_imp_words,Y,lr)
clf = lr_func(X, Y_train)
manual_check(df_with_imp_words,Y,clf,200)
#coef_plot(clf,X)
