import newspaper
import goose3
import pandas as pd
import nltk.data
import re
from nltk.corpus import brown
import nltk
import json

import time
startTime = time.time()
print('start: ', startTime )

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
# originalDataFrame = originalDataFrame.iloc[0:6]  # filters 50 first rows of data frame, for quick and easy self checking.

# irrelevant - just a way to check the reading of CSV file. printing the data frame and its columns.
# print(originalDataFrame)
'''print(originalDataFrame.columns)
for column in originalDataFrame.columns:
    print( column, ": ", originalDataFrame.at[4046, column])'''


# now we want to create some kind of 'our' data frame. we want to fill it with data collected from goose and newspaper
# libraries. Then, we want to self (and manually) check if the filled data is correct and corresponding the actual
# websites.
# ourDataFrame = pd.DataFrame() # create the empty df we want to fill
# ourDataFrame['URLs'] = 0 # all following 5 rows sort of create the columns of the df.
# ourDataFrame['Author'] = []
# ourDataFrame['Date'] = 0
# ourDataFrame['HeadLine'] = 0
# ourDataFrame['Body'] = 0
#
# count = 0  # simply a way to access index when filling data in our df, it avoids errors (notice at the end of code when it's used)
# for index, row in originalDataFrame.iterrows():  # iterate the df
#     gooseErrorFlag = False # flag stating is the goose article was successfully extracted.
#     currURL = originalDataFrame.at[index, 'URLs']
#     currURL = currURL.replace('\r\n', '') # remove the "\r\n" at the end of URL, it avoids errors.
#
#     # irrelevant part of code, it used for self checking a particular article, keep it for now.
#     '''if 'comedy-speaks-to-modern-america-says' in currURL:
#         print(currURL)
#         cleanURL = currURL.replace('\r\n', '')
#         breakPoint=1'''
#
#     # filling newspaper parameters.
#     newspaperArticle = newspaper.Article(currURL)
#     newspaperArticle.download()
#     if newspaperArticle.download_state == 2:  # checks validation of downloading: 2=downloaded, 1=unsuccessful download
#         newspaperArticle.parse()
#         newspaperAuthor = newspaperArticle.authors
#         newspaperDate = newspaperArticle.publish_date
#         newspaperHeadLine = newspaperArticle.title
#         newspaperBody = newspaperArticle.text
#     else: # if article couldn't download, enter None to all parameters.
#         newspaperAuthor = None
#         newspaperDate = None
#         newspaperHeadLine = None
#         newspaperBody = None
#
#     # filling goose parameters.
#     goose = goose3.Goose()
#     try:
#         gooseArticle = goose.extract(url=currURL)  # trying to extract the URL
#     except Exception as error:  # code enters the exception part only if trying wasn't successful.
#         errorString = str(error)  # regular error is not iterable, thus convert to string.
#         if '404' in errorString:  # if error occurred, fill in Nones + raise the flag to True.
#             gooseAuthor = None
#             gooseDate = None
#             gooseHeadLine = None
#             gooseBody = None
#             gooseErrorFlag = True
#
#     if gooseErrorFlag == False:  # if no errors, fill in the parameters.
#         gooseAuthor = gooseArticle.authors
#         gooseDate = gooseArticle.publish_date
#         gooseHeadLine = gooseArticle.title
#         gooseBody = gooseArticle.cleaned_text
#
#     # fill in our df with one of the libraries parameters, newspaper is chosen here arbitrary.
#     ourDataFrame.loc[count, 'URLs'] = currURL
#     ourDataFrame.loc[count, 'Author'] = str(newspaperAuthor) # df can't have list as a value, thus convert to string.
#     ourDataFrame.loc[count, 'Date'] = newspaperDate
#     ourDataFrame.loc[count, 'HeadLine'] = newspaperHeadLine
#     ourDataFrame.loc[count, 'Body'] = newspaperBody
#
#     # following 4 IFs: if one of the parameters is empty or None, replace it with the relevant data from the second
#     # library, as it might have better content.
#     if ( (ourDataFrame.loc[count, 'Author']==None)  or (ourDataFrame.loc[count, 'Author']=='[]') ):
#         ourDataFrame.loc[count, 'Author'] = str(gooseAuthor)
#
#     if (ourDataFrame.loc[count, 'Date'] == None):
#         ourDataFrame.loc[count, 'Date'] = gooseDate
#
#     if ((ourDataFrame.loc[count, 'HeadLine'] == None) or (ourDataFrame.loc[count, 'HeadLine'] == '')):
#         ourDataFrame.loc[count, 'HeadLine'] = gooseHeadLine
#
#     if ((ourDataFrame.loc[count, 'Body'] == None) or (ourDataFrame.loc[count, 'Body'] == '')):
#         ourDataFrame.loc[count, 'Body'] = gooseBody
#
#     count = count + 1
#
# breakPoint=0


# -----------------------------------------------------------------------------

# checks if there's a typo or a spelling mistake in a single world
# d = enchant.Dict("en_US")
# d.check("enchant")


headerFlagError = False
bodyFlagError = False
initDataPerArticle = {}
for index, row in originalDataFrame.iterrows():
    articleAttrs = {}
    headerFlagError = False
    bodyFlagError = False
    currentURL = originalDataFrame.at[index, 'URLs']
    currentHeadLineList = originalDataFrame.at[index, 'Headline']
    currentBodyList = originalDataFrame.at[index, 'Body']
    if ((currentHeadLineList == []) or (not(isinstance(currentHeadLineList, str)))):
        headerFlagError = True
    if ((currentBodyList == []) or (not(isinstance(currentBodyList, str)))):
        bodyFlagError = True

    if ( (headerFlagError==False) and (bodyFlagError==False) ):
        currentHeadLineStr = re.sub("[^\w]", " ",  currentHeadLineList).split()
        currentBodyStr = re.sub("[^\w]", " ",  currentBodyList).split()

        # number of sentences, mean length, shortest and longest ones
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        HeadLineSentences = tokenizer.tokenize(currentHeadLineList)
        BodySentences = tokenizer.tokenize(currentBodyList)

        articleAttrs['numOfHeaderSentences'] = len(HeadLineSentences)
        articleAttrs['numOfBodySentences'] = len(BodySentences)

        meanSentencesCount = 0
        minHeaderLength = 999999999
        maxHeaderLength = 0
        for headerSentence in HeadLineSentences:
            words = re.sub("[^\w]", " ",  headerSentence).split()
            sumToAdd = len(words)
            meanSentencesCount = meanSentencesCount + sumToAdd

            if len(words) != 0:
                if len(words) < minHeaderLength:
                    minHeaderLength = len(words)

                if len(words) > maxHeaderLength:
                    maxHeaderLength = len(words)

        articleAttrs['meanHeaderLen'] = meanSentencesCount / len(HeadLineSentences)
        articleAttrs['minHeaderLength'] = minHeaderLength
        articleAttrs['maxHeaderLength'] = maxHeaderLength


        meanSentencesCount = 0
        minBodyLength = 999999999
        maxBodyLength = 0
        for bodySentence in BodySentences:
            words = re.sub("[^\w]", " ",  bodySentence).split()
            sumToAdd = len(words)
            meanSentencesCount = meanSentencesCount + sumToAdd

            if len(words) != 0:
                if len(words) < minBodyLength:
                    minBodyLength = len(words)

                if len(words) > maxBodyLength:
                    maxBodyLength = len(words)

        articleAttrs['meanBodyLen'] = meanSentencesCount / len(BodySentences)
        articleAttrs['minBodyLength'] = minBodyLength
        articleAttrs['maxBodyLength'] = maxBodyLength


        # check for misspells
        word_list = brown.words()
        word_set = set(word_list)
        headerMisspellCount = 0
        bodyMisspellCount = 0
        for headerWord in currentHeadLineStr:
            if headerWord not in word_set:
                headerMisspellCount = headerMisspellCount + 1

        for bodyWord in currentBodyStr:
            if bodyWord not in word_set:
                bodyMisspellCount = bodyMisspellCount + 1

        articleAttrs['headerMisspellRate'] = headerMisspellCount / len(currentHeadLineStr)
        articleAttrs['bodyMisspellRate'] = bodyMisspellCount / len(currentBodyStr)


        # check for word grammar category
        headerGrammarDict = {}
        for line in HeadLineSentences:
            tmp = nltk.word_tokenize(line)
            grammarPerLine = nltk.pos_tag(tmp)

            for pair in grammarPerLine:
                grammarType = pair[1]

                if grammarType not in headerGrammarDict.keys():
                    headerGrammarDict[grammarType] = 1
                else:
                    tmpValue = headerGrammarDict[grammarType]
                    headerGrammarDict[grammarType] = tmpValue + 1

        bodyGrammarDict = {}
        for line in BodySentences:
            tmp = nltk.word_tokenize(line)
            grammarPerLine = nltk.pos_tag(tmp)

            for pair in grammarPerLine:
                grammarType = pair[1]

                if grammarType not in bodyGrammarDict.keys():
                    bodyGrammarDict[grammarType] = 1
                else:
                    tmpValue = bodyGrammarDict[grammarType]
                    bodyGrammarDict[grammarType] = tmpValue + 1

        headerGrammarMaxKey = max(headerGrammarDict.keys(), key=(lambda k: headerGrammarDict[k]))
        headerGrammarMinKey = min(headerGrammarDict.keys(), key=(lambda k: headerGrammarDict[k]))
        headerGrammarMaxTimes = headerGrammarDict[headerGrammarMaxKey]
        headerGrammarMinTimes = headerGrammarDict[headerGrammarMinKey]
        articleAttrs['headerGrammarMaxKey'] = headerGrammarMaxKey
        articleAttrs['headerGrammarMaxTimes'] = headerGrammarMaxTimes
        articleAttrs['headerGrammarMinKey'] = headerGrammarMinKey
        articleAttrs['headerGrammarMinTimes'] = headerGrammarMinTimes

        bodyGrammarMaxKey = max(bodyGrammarDict.keys(), key=(lambda k: bodyGrammarDict[k]))
        bodyGrammarMinKey = min(bodyGrammarDict.keys(), key=(lambda k: bodyGrammarDict[k]))
        bodyGrammarMaxTimes = bodyGrammarDict[bodyGrammarMaxKey]
        bodyGrammarMinTimes = bodyGrammarDict[bodyGrammarMinKey]
        articleAttrs['bodyGrammarMaxKey'] = bodyGrammarMaxKey
        articleAttrs['bodyGrammarMaxTimes'] = bodyGrammarMaxTimes
        articleAttrs['bodyGrammarMinKey'] = bodyGrammarMinKey
        articleAttrs['bodyGrammarMinTimes'] = bodyGrammarMinTimes

    else:
        articleAttrs['numOfHeaderSentences'] = 0
        articleAttrs['numOfBodySentences'] = 0
        articleAttrs['meanHeaderLen'] = 0
        articleAttrs['minHeaderLength'] = 0
        articleAttrs['maxHeaderLength'] = 0
        articleAttrs['meanBodyLen'] = 0
        articleAttrs['minBodyLength'] = 0
        articleAttrs['maxBodyLength'] = 0
        articleAttrs['headerMisspellRate'] = 0
        articleAttrs['bodyMisspellRate'] = 0
        articleAttrs['headerGrammarMaxKey'] = 0
        articleAttrs['headerGrammarMaxTimes'] = 0
        articleAttrs['headerGrammarMinKey'] = 0
        articleAttrs['headerGrammarMinTimes'] = 0
        articleAttrs['bodyGrammarMaxKey'] = 0
        articleAttrs['bodyGrammarMaxTimes'] = 0
        articleAttrs['bodyGrammarMinKey'] = 0
        articleAttrs['bodyGrammarMinTimes'] = 0

    initDataPerArticle[currentURL] = articleAttrs


# print(initDataPerArticle)

attrsDataFrame = pd.DataFrame()
attrsDataFrame['URL'] = 0
attrsDataFrame['numOfHeaderSentences'] = 0
attrsDataFrame['numOfBodySentences'] = 0
attrsDataFrame['meanHeaderLen'] = 0
attrsDataFrame['minHeaderLength'] = 0
attrsDataFrame['maxHeaderLength'] = 0
attrsDataFrame['meanBodyLen'] = 0
attrsDataFrame['minBodyLength'] = 0
attrsDataFrame['maxBodyLength'] = 0
attrsDataFrame['headerMisspellRate'] = 0
attrsDataFrame['bodyMisspellRate'] = 0
attrsDataFrame['headerGrammarMaxKey'] = 0
attrsDataFrame['headerGrammarMaxTimes'] = 0
attrsDataFrame['headerGrammarMinKey'] = 0
attrsDataFrame['headerGrammarMinTimes'] = 0
attrsDataFrame['bodyGrammarMaxKey'] = 0
attrsDataFrame['bodyGrammarMaxTimes'] = 0
attrsDataFrame['bodyGrammarMinKey'] = 0
attrsDataFrame['bodyGrammarMinTimes'] = 0

count = 0
for url, attrs in initDataPerArticle.items():
    attrsDataFrame.loc[count, 'URL'] = url
    attrsDataFrame.loc[count, 'numOfHeaderSentences'] = attrs['numOfHeaderSentences']
    attrsDataFrame.loc[count, 'numOfBodySentences'] = attrs['numOfBodySentences']
    attrsDataFrame.loc[count, 'meanHeaderLen'] = attrs['meanHeaderLen']
    attrsDataFrame.loc[count, 'minHeaderLength'] = attrs['minHeaderLength']
    attrsDataFrame.loc[count, 'maxHeaderLength'] = attrs['maxHeaderLength']
    attrsDataFrame.loc[count, 'meanBodyLen'] = attrs['meanBodyLen']
    attrsDataFrame.loc[count, 'minBodyLength'] = attrs['minBodyLength']
    attrsDataFrame.loc[count, 'maxBodyLength'] = attrs['maxBodyLength']
    attrsDataFrame.loc[count, 'headerMisspellRate'] = attrs['headerMisspellRate']
    attrsDataFrame.loc[count, 'bodyMisspellRate'] = attrs['bodyMisspellRate']
    attrsDataFrame.loc[count, 'headerGrammarMaxKey'] = attrs['headerGrammarMaxKey']
    attrsDataFrame.loc[count, 'headerGrammarMaxTimes'] = attrs['headerGrammarMaxTimes']
    attrsDataFrame.loc[count, 'headerGrammarMinKey'] = attrs['headerGrammarMinKey']
    attrsDataFrame.loc[count, 'headerGrammarMinTimes'] = attrs['headerGrammarMinTimes']
    attrsDataFrame.loc[count, 'bodyGrammarMaxKey'] = attrs['bodyGrammarMaxKey']
    attrsDataFrame.loc[count, 'bodyGrammarMaxTimes'] = attrs['bodyGrammarMaxTimes']
    attrsDataFrame.loc[count, 'bodyGrammarMinKey'] = attrs['bodyGrammarMinKey']
    attrsDataFrame.loc[count, 'bodyGrammarMinTimes'] = attrs['bodyGrammarMinTimes']

    count = count + 1

attrsDataFrame.to_csv('initDataPerArticle.csv', encoding='utf-8', index=False)
import time
print('finish: ', time.time() - startTime )









''''# takes a string of body text and splits (and prints) it into whole sentences.
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
print( '\n-----\n'.join(tokenizer.tokenize(originalDataFrame.at[0, 'Body'])))


word_list = brown.words()
word_set = set(word_list)
print('hello' in word_set)
print('hellos' in word_set)


with open('alonFakeData.txt') as f:
    for line in f:
        tmp = nltk.word_tokenize(line)
        print (nltk.pos_tag(tmp))'''







