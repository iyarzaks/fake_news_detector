import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydot
from sklearn.feature_extraction.text import TfidfTransformer
import graphviz
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold

def compare_clfs(X,Y,X_TDF,Y_TDF):
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA","Logistic regression"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(hidden_layer_sizes=(100, ), activation="relu", solver="adam", alpha=0.0001, batch_size="auto", learning_rate="constant", learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),LogisticRegressionCV(Cs=10
            ,penalty='l2'
            ,scoring='accuracy'
            ,cv=10
            ,random_state=777
            ,max_iter=10000
            ,fit_intercept=True
            ,solver='newton-cg'
            ,tol=10
        )]
    results_list=[]
    cv_k_fold = KFold (n_splits=10,shuffle=True)
    for clf in zip(names,classifiers):
        scores = cross_val_score(clf[1], X, Y, cv=cv_k_fold)
        results_list.append(np.mean(scores))
        print(clf[0],np.mean(scores))
    for clf in zip(names, classifiers):
        scores = cross_val_score(clf[1], X_TDF, Y_TDF, cv=cv_k_fold)
        results_list.append(np.mean(scores))
        print(clf[0]+"_tdf_idf", np.mean(scores))
    new_list=[]
    for name in names:
        new_list.append(name+"_tf_idf")
    names = names+new_list
    results_series = pd.Series(results_list, index=names)
    results_series = results_series.sort_values()
    results_series.sort_index().plot(kind="bar")
    import matplotlib.pyplot
    matplotlib.pyplot.show()



def to_bag_of_words(articles,labels,top_50_path,importance_path,num_of_words):
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
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    res = dict(zip(count_vect.get_feature_names(),
               mutual_info_classif(X_train_counts, labels, discrete_features=True)
               ))
    #print (res)
    # print (X_train_counts.shape)
    # print (X_train_counts[:,count_vect.vocabulary_['www']])
    sorted_d = sorted(res.items(), key=lambda x: -x[1])
    top_50_words = [k for (k,v) in sorted_d[:num_of_words]]
    print_to_json(top_50_words,top_50_path)
    print_to_json(sorted_d, importance_path)
    #print (count_vect.vocabulary_)
    #print(count_vect.vocabulary_.get('algorithm'))


def build_table_form_words(data,words,option="r"):
    data.rename(index=str, columns={"text": "Body"})
    articles_list = data['Body'].values.astype('U').tolist()
    count_vect = CountVectorizer(max_df=0.95, min_df=2,
                                     max_features=50000,
                                     stop_words='english')
    dtm = count_vect.fit_transform(articles_list)
    if option == "tf_idf":
        tfidf_transformer = TfidfTransformer()
        dtm = tfidf_transformer.fit_transform(dtm)
    df = pd.DataFrame(dtm.toarray(), columns=count_vect.get_feature_names())
    df = pd.DataFrame (df ,columns=words)
    df.index = range(len(df))
    data.index = range(len(df))
    result = pd.concat([data, df],axis=1)
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


def coef_plot(lr,X):
    import matplotlib.pyplot
    coef_list = lr.coef_[0].tolist()
    coef_list = [float(format(a, '.2f')) for a in coef_list]
    coefs1_series = pd.Series(coef_list, index=list(X.columns.values))
    coefs1_series.sort_values().plot(kind="barh")
    matplotlib.pyplot.show()


def manual_check(df_with_imp_words,Y,lr,num_of_words):
    X_test = df_with_imp_words.iloc[:,6:]
    Y_test = Y.iloc[:]
    #print (X_test.to_dict('records'))
    pairs = zip(lr.predict(X_test),Y_test)
    un_matched_digits = [(idx,pair) for idx, pair in enumerate(pairs) if pair[0]!=str(int(pair[1]))]
    print (un_matched_digits)
    print (1-len(un_matched_digits)/len(Y_test))


def nn_func(X,Y):
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", alpha=0.0001, batch_size="auto",
                  learning_rate="constant", learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                  random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                  early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    clf = clf.fit(X, Y)
    return clf
    #scores = lr.scores_['1']
    #mean_scores = np.mean(scores, axis=0)



def lr_func(X,Y):
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
    lr = lr.fit(X, Y)
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


def dec_tree_func(X, Y_train):
    clf = DecisionTreeClassifier(min_impurity_decrease=0.01)
    clf.fit(X, Y_train)
    scores = cross_val_score(clf, X, Y_train, cv=10)
    print (np.mean(scores))
    tree.export_graphviz(clf,out_file = 'tree.dot')
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    #graph.draw('file.png')
    #graph.write_png('tree.png')
    return clf