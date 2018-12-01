import numpy as np
import pandas as pd
from project_utils import *
from sklearn import linear_model, datasets
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegressionCV
import sys
#from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import subprocess
import os
os.environ["PATH"] += 'C:/Users/win7/AppData/Local/Programs/Python\Python36-32/Lib/site-packages/graphviz/'

def flow():
    #all_data = pd.read_csv('fake_or_real_news.csv', engine='python',usecols=["URLs","Headline","Body","Label (1=true)","numOfHeaderSentences","meanHeaderLen","headerMisspellRate"])
    all_data = pd.read_csv('Classified_Data_kaggle.csv', engine='python',nrows =4000)
    print (all_data.shape)
    # all_data_csv_2 = pd.read_csv('Classified_Data_kaggle.csv', engine='python',nrows=2000)
    all_data = all_data.replace("FAKE", '0')
    all_data = all_data.replace("REAL", '1')
    all_data = all_data.rename(index=str, columns={"text": "Body","label":'Label (1=true)'})
    # all_data_csv_2 =all_data_csv_2[["Body","Label (1=true)"]]
    # all_data = all_data[["Body", "Label (1=true)"]]
    # all_data = pd.concat([all_data,all_data_csv_2],axis=0)
    bad_input = all_data.loc[~all_data['Label (1=true)'].isin(['0','1'])]
    print_to_json(bad_input.index.values.tolist(),"bad_input_indexes")
    data = all_data.loc[all_data['Label (1=true)'].isin(['0','1'])]
    print (data.shape)
    # articles_list = data['Body'].values.astype('U').tolist()
    # labels = data['Label (1=true)'].values.astype('U').tolist()
    # to_bag_of_words(articles_list, labels, "top_50.json.1","words_importance.json.1",500)
    words = read_json("top_50.json.1")
    #words_tf_idf = read_json("top_50.tfidf.json")
    #to_bag_of_words(artcles_list,labels)
    words = words[:500]
    #words_tf_idf = words_tf_idf[:100]
    #df_with_imp_words_tf_idf = build_table_form_words(data, words_tf_idf,option="tf_idf")
    df_with_imp_words = build_table_form_words(data,words)
    df_with_imp_words = df_with_imp_words.replace(np.nan, 0)
    #df_with_imp_words_tf_idf = df_with_imp_words_tf_idf.replace(np.nan, 0)
    X = df_with_imp_words.iloc[:,6:]
    #X_tdf = df_with_imp_words_tf_idf.iloc[:,6:]
    Y_train = df_with_imp_words['Label (1=true)']
    #Y_tdf = df_with_imp_words_tf_idf['Label (1=true)']

    #fold = KFold(len(Y_train), n_folds=10, shuffle=True, random_state=777)
    #lr = lr_func(X,Y_train)


    #mean_scores = np.mean(scores, axis=0)
    # coef_plot(lr,df_with_imp_words)

    #manual_check(df_with_imp_words,Y,lr)
    nn_clf = nn_func(X, Y_train)
    adaboost_clf = adaboost(X, Y_train)
    lr_clf = lr_func(X, Y_train)
    dec_tree_clf = dec_tree_func(X, Y_train)
    r_forest_clf = r_forest(X,Y_train)

    #coef_plot(clf, X)
    # save the classifier
    joblib.dump(nn_clf, 'nn_clf.pkl')
    joblib.dump(adaboost_clf, 'adaboost_clf.pkl')
    joblib.dump(lr_clf, 'lr_clf.pkl')
    joblib.dump(dec_tree_clf, 'dec_tree_clf.pkl')
    joblib.dump(r_forest_clf, 'r_forest_clf.pkl')

# load it again
# clf_from_file = joblib.load('lr_clf.pkl')
# manual_check(df_with_imp_words,Y,clf_from_file,200)
#coef_plot(clf,X)

def check_new_df():
    nn_clf_from_file = joblib.load('nn_clf.pkl')
    lr_clf_from_file = joblib.load('nn_clf.pkl')
    test_df = pd.read_csv('Classified_Data_kaggle.csv', engine='python', skiprows=list(range(1,3000)),nrows=1000)
    test_df = test_df.replace("FAKE", '0')
    test_df = test_df.replace("REAL", '1')
    test_df = test_df.rename(index=str, columns={"text": "Body","label":'Label (1=true)'})
    # test_df.loc[test_df["label"] == "REAL"]["label"] = '1'
    words = read_json("top_50.json.1")
    words = words[:100]
    df_with_imp_words = build_table_form_words(test_df, words)
    # print (df_with_imp_words.iloc[[1],:200].values)
    df_with_imp_words = df_with_imp_words.dropna(thresh=4)
    df_with_imp_words = df_with_imp_words.replace(np.nan, 0)
    #df_with_imp_words = df_with_imp_words.iloc[0:1000,:]
    Y = df_with_imp_words['Label (1=true)']
    #print (Y)
    manual_check(df_with_imp_words, Y, nn_clf_from_file, 100)


def main():
    flow()
    #check_new_df()

if __name__ == "__main__":
    main()
