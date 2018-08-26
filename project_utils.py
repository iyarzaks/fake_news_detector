import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
def to_bag_of_words(articles):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(articles)
    print(vectorizer.vocabulary_)
    print(vectorizer.idf_)
    vector = vectorizer.transform([articles[0]])
    print(vector.shape)
    print(vector.toarray())