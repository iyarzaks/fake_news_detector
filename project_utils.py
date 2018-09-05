import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def to_bag_of_words(articles,labels):
    # vectorizer = TfidfVectorizer()
    # vectorizer.fit(articles)
    # print(vectorizer.vocabulary_)
    # print(vectorizer.idf_)
    # vector = vectorizer.transform([articles[0]])
    # print(vector.shape)
    # print(vector.toarray())
    categories = [0, 1]

    count_vect = CountVectorizer(max_df=0.95, min_df=2,
                                     max_features=50000,
                                     stop_words='english')
    X_train_counts = count_vect.fit_transform(articles)
    res = dict(zip(count_vect.get_feature_names(),
               mutual_info_classif(X_train_counts, labels, discrete_features=True)
               ))
    #print (res)
    print (X_train_counts.shape)
    print (X_train_counts[:,count_vect.vocabulary_['www']])
    sorted_d = sorted(res.items(), key=lambda x: -x[1])
    top_50_words = [k for (k,v) in sorted_d[:50]]
    print (top_50_words)
    print_to_json(top_50_words,"top_50.json")
    print_to_json(sorted_d, "words_importance.json")
    #print (count_vect.vocabulary_)
    #print(count_vect.vocabulary_.get('algorithm'))


def build_table_form_words(data,words):
    articles_list = data['Body'].values.astype('U').tolist()
    count_vect = CountVectorizer(max_df=0.95, min_df=2,
                                 max_features=50000,
                                 stop_words='english')
    dtm = count_vect.fit_transform(articles_list)
    df = pd.DataFrame(dtm.toarray(), columns=count_vect.get_feature_names())
    df = pd.DataFrame (df ,columns=words)
    result = pd.concat([df, data], axis =1)
    return result


    #filterd = df.loc[df[df.index.name].get_feature_name().isin(words)]
    #print (filterd.shape)

def print_to_json(data, path):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)


def read_json(name):
    with open(name) as inputfile:
        data = json.load(inputfile)
        return data


def coef_plot(lr,df_with_imp_words):
    import matplotlib.pyplot
    coef_list = lr.coef_[0].tolist()
    coef_list = [float(format(a, '.2f')) for a in coef_list]
    coefs1_series = pd.Series(coef_list, index=list(df_with_imp_words.columns.values)[:50])
    coefs1_series.sort_values().plot(kind="bar")
    matplotlib.pyplot.show()

def manual_check(df_with_imp_words,Y,lr):
    X_test = df_with_imp_words.iloc[3000:4000,0:50]
    Y_test = Y.iloc[3000:4000]
    pairs = zip(lr.predict(X_test),Y_test)
    un_matched_digits = [(idx,pair) for idx, pair in enumerate(pairs) if pair[0] != pair[1]]
    print (un_matched_digits)
    print (1-len(un_matched_digits)/len(Y_test))


def lr_func(X,Y_train):
    lr = LogisticRegressionCV(Cs=10
            ,penalty='l2'
            ,scoring='accuracy'
            ,cv=10
            ,random_state=777
            ,max_iter=10000
            ,fit_intercept=True
            ,solver='newton-cg'
            ,tol=10
        )
    lr = lr.fit(X, Y_train)
    print ('Max auc:', lr.scores_['1'].mean(axis=0).max())
    return lr
    #scores = lr.scores_['1']
    #mean_scores = np.mean(scores, axis=0)

def svm_func(X, Y_train):
    clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    clf.fit(X, Y_train)
    scores = cross_val_score(clf, X, Y_train, cv=10)
    print (np.mean(scores))
    return clf