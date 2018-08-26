import pandas
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
def to_bag_of_words(articles):
    # vectorizer = TfidfVectorizer()
    # vectorizer.fit(articles)
    # print(vectorizer.vocabulary_)
    # print(vectorizer.idf_)
    # vector = vectorizer.transform([articles[0]])
    # print(vector.shape)
    # print(vector.toarray())
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(articles)
    print (X_train_counts.shape)
    print (X_train_counts[:,count_vect.vocabulary_['and']])
    print (count_vect.vocabulary_)