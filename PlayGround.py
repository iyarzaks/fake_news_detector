import newspaper
import goose3
import pandas as pd

myURL = "https://www.reuters.com/article/us-filmfestival-london-lastflagflying/linklaters-war-veteran-comedy-speaks-to-modern-america-says-star-idUSKBN1CD0X2"

article = newspaper.Article(myURL)
article.download()
article.parse()
print("Authors: ", article.authors)
print("Date: ", article.publish_date)
print("Headline: ", article.title)
print("Body: ", article.text)

