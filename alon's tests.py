import newspaper
import goose3
import pandas as pd

# myURL = "http://www.bbc.com/news/world-us-canada-41419190" # irrelevant
# myURL = "https://www.nytimes.com/2017/10/04/us/marilou-danley-stephen-paddock.html" # irrelevant

# the following code is just an example of using 'newspaper' library.
'''article = newspaper.Article(myURL)
article.download()
article.parse()
print("Authors: ", article.authors)
print("Date: ", article.publish_date)
print("Headline: ", article.title)
print("Body: ", article.text)'''

# the following code is just an example of using 'goose' library.
'''g = goose3.Goose()
myArticle = g.extract(url=myURL)
print("Authors: ", myArticle.authors)
print("Date: ", myArticle.publish_date)
print("Headline: ", myArticle.title)
print("Body: ", myArticle.cleaned_text)'''


originalDataFrame = pd.read_csv('Classified_Data_kaggle.csv', encoding = "ISO-8859-1")  # reading CSV file of data, ISO encoding is just a standard way to avoid some errors.
originalDataFrame = originalDataFrame.iloc[0:50]  # filters 50 first rows of data frame, for quick and easy self checking.

# irrelevant - just a way to check the reading of CSV file. printing the data frame and its columns.
# print(originalDataFrame)
'''print(originalDataFrame.columns)
for column in originalDataFrame.columns:
    print( column, ": ", originalDataFrame.at[4046, column])'''


# now we want to create some kind of 'our' data frame. we want to fill it with data collected from goose and newspaper
# libraries. Then, we want to self (and manually) check if the filled data is correct and corresponding the actual
# websites.
ourDataFrame = pd.DataFrame() # create the empty df we want to fill
ourDataFrame['URLs'] = 0 # all following 5 rows sort of create the columns of the df.
ourDataFrame['Author'] = []
ourDataFrame['Date'] = 0
ourDataFrame['HeadLine'] = 0
ourDataFrame['Body'] = 0

count = 0  # simply a way to access index when filling data in our df, it avoids errors (notice at the end of code when it's used)
for index, row in originalDataFrame.iterrows():  # iterate the df
    gooseErrorFlag = False # flag stating is the goose article was successfully extracted.
    currURL = originalDataFrame.at[index, 'URLs']
    currURL = currURL.replace('\r\n', '') # remove the "\r\n" at the end of URL, it avoids errors.

    # irrelevant part of code, it used for self checking a particular article, keep it for now.
    '''if 'comedy-speaks-to-modern-america-says' in currURL:
        print(currURL)
        cleanURL = currURL.replace('\r\n', '')
        breakPoint=1'''

    # filling newspaper parameters.
    newspaperArticle = newspaper.Article(currURL)
    newspaperArticle.download()
    if newspaperArticle.download_state == 2:  # checks validation of downloading: 2=downloaded, 1=unsuccessful download
        newspaperArticle.parse()
        newspaperAuthor = newspaperArticle.authors
        newspaperDate = newspaperArticle.publish_date
        newspaperHeadLine = newspaperArticle.title
        newspaperBody = newspaperArticle.text
    else: # if article couldn't download, enter None to all parameters.
        newspaperAuthor = None
        newspaperDate = None
        newspaperHeadLine = None
        newspaperBody = None

    # filling goose parameters.
    goose = goose3.Goose()
    try:
        gooseArticle = goose.extract(url=currURL)  # trying to extract the URL
    except Exception as error:  # code enters the exception part only if trying wasn't successful.
        errorString = str(error)  # regular error is not iterable, thus convert to string.
        if '404' in errorString:  # if error occurred, fill in Nones + raise the flag to True.
            gooseAuthor = None
            gooseDate = None
            gooseHeadLine = None
            gooseBody = None
            gooseErrorFlag = True

    if gooseErrorFlag == False:  # if no errors, fill in the parameters.
        gooseAuthor = gooseArticle.authors
        gooseDate = gooseArticle.publish_date
        gooseHeadLine = gooseArticle.title
        gooseBody = gooseArticle.cleaned_text

    # fill in our df with one of the libraries parameters, newspaper is chosen here arbitrary.
    ourDataFrame.loc[count, 'URLs'] = currURL
    ourDataFrame.loc[count, 'Author'] = str(newspaperAuthor) # df can't have list as a value, thus convert to string.
    ourDataFrame.loc[count, 'Date'] = newspaperDate
    ourDataFrame.loc[count, 'HeadLine'] = newspaperHeadLine
    ourDataFrame.loc[count, 'Body'] = newspaperBody

    # following 4 IFs: if one of the parameters is empty or None, replace it with the relevant data from the second
    # library, as it might have better content.
    if ( (ourDataFrame.loc[count, 'Author']==None)  or (ourDataFrame.loc[count, 'Author']=='[]') ):
        ourDataFrame.loc[count, 'Author'] = str(gooseAuthor)

    if (ourDataFrame.loc[count, 'Date'] == None):
        ourDataFrame.loc[count, 'Date'] = gooseDate

    if ((ourDataFrame.loc[count, 'HeadLine'] == None) or (ourDataFrame.loc[count, 'HeadLine'] == '')):
        ourDataFrame.loc[count, 'HeadLine'] = gooseHeadLine

    if ((ourDataFrame.loc[count, 'Body'] == None) or (ourDataFrame.loc[count, 'Body'] == '')):
        ourDataFrame.loc[count, 'Body'] = gooseBody

    count = count + 1

breakPoint=0













