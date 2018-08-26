import pandas
import json
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_selection import mutual_info_classif
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
    sorted_d = sorted(res.items(), key=lambda x: x[1])
    print_to_json(sorted_d, "words_importance.json")
    #print (count_vect.vocabulary_)
    #print(count_vect.vocabulary_.get('algorithm'))


def print_to_json(data, path):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)