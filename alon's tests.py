import newspaper
import goose3
import pandas as pd

# myURL = "http://www.bbc.com/news/world-us-canada-41419190"
myURL = "https://www.nytimes.com/2017/10/04/us/marilou-danley-stephen-paddock.html"

'''article = newspaper.Article(myURL)
article.download()
article.parse()
print("Authors: ", article.authors)
print("Date: ", article.publish_date)
print("Headline: ", article.title)
print("Body: ", article.text)


g = goose3.Goose()
myArticle = g.extract(url=myURL)
print("Authors: ", myArticle.authors)
print("Date: ", myArticle.publish_date)
print("Headline: ", myArticle.title)
print("Body: ", myArticle.cleaned_text)'''

myArticlesDict = {}

originalDataFrame = pd.read_csv('Classified_Data_kaggle.csv', encoding = "ISO-8859-1")
# print(originalDataFrame)
'''print(originalDataFrame.columns)
for column in originalDataFrame.columns:
    print( column, ": ", originalDataFrame.at[4046, column])'''

ourDataFrame = pd.DataFrame()
ourDataFrame['URLs'] = 0
ourDataFrame['Author'] = 0
ourDataFrame['Date'] = 0
ourDataFrame['HeadLine'] = 0
ourDataFrame['Body'] = 0

count = 0
for index, row in originalDataFrame.iterrows():
    gooseErrorFlag = False
    currURL = originalDataFrame.at[index, 'URLs']

    if 'comedy-speaks-to-modern-america-says' in currURL:
        breakPoint=1

    newspaperArticle = newspaper.Article(currURL)
    newspaperArticle.download()
    if newspaperArticle.download_state == 2: # 2=downloaded, 1=unsuccessful download
        newspaperArticle.parse()
        newspaperAuthor = newspaperArticle.authors
        newspaperDate = newspaperArticle.publish_date
        newspaperHeadLine = newspaperArticle.title
        newspaperBody = newspaperArticle.text
    else:
        newspaperAuthor = None
        newspaperDate = None
        newspaperHeadLine = None
        newspaperBody = None

    # goose = goose3.Goose( {'strict':False} )
    goose = goose3.Goose()
    try:
        gooseArticle = goose.extract(url=currURL)
    except Exception as error:
        errorString = str(error)
        if '404' in errorString:
            gooseAuthor = None
            gooseDate = None
            gooseHeadLine = None
            gooseBody = None
            gooseErrorFlag = True

    if gooseErrorFlag == False:
        gooseAuthor = gooseArticle.authors
        gooseDate = gooseArticle.publish_date
        gooseHeadLine = gooseArticle.title
        gooseBody = gooseArticle.cleaned_text

    ourDataFrame.loc[count, 'URLs'] = currURL
    '''ourDataFrame.loc[count, 'Author'] = newspaperAuthor
    ourDataFrame.loc[count, 'Date'] = newspaperDate
    ourDataFrame.loc[count, 'HeadLine'] = newspaperHeadLine
    ourDataFrame.loc[count, 'Body'] = newspaperBody'''

    '''if ( (ourDataFrame.loc[count, 'Author']==None) or (ourDataFrame.loc[count, 'Author']=='None') or (ourDataFrame.loc[count, 'Author']=='')  ):
        ourDataFrame.loc[count, 'Author'] = gooseAuthor
        x=0'''

    count = count + 1

    #print(ourDataFrame.loc[[0]])
    x=0













