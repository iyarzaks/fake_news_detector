import panda
from sklearn.feature_extraction.text import TfidfVectorizer
def to_bag_of_words(articles):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(articles)
    print(vectorizer.vocabulary_)