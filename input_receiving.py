import newspaper
import goose3
import pandas as pd
import nltk.data
import re
from nltk.corpus import brown
import nltk
import json
from copy import deepcopy


def convertUrlToDF(inputURL):
    # create an empty df we want to fill
    singleInputDF = pd.DataFrame()
    singleInputDF.loc[0, 'URLs'] = inputURL
    singleInputDF['HeadLine'] = 0
    singleInputDF['Body'] = 0

    gooseErrorFlag = False  # flag stating if the goose article was successfully extracted.

    # filling newspaper parameters.
    newspaperArticle = newspaper.Article(inputURL)
    newspaperArticle.download()
    if newspaperArticle.download_state == 2:  # checks validation of downloading: 2=downloaded, 1=unsuccessful download
        newspaperArticle.parse()
        newspaperHeadLine = newspaperArticle.title
        newspaperBody = newspaperArticle.text
    else:  # if article couldn't download, enter None to all parameters.
        newspaperHeadLine = None
        newspaperBody = None

    # filling goose parameters.
    goose = goose3.Goose()
    try:
        gooseArticle = goose.extract(url=inputURL)  # trying to extract the URL
    except Exception as gooseError:  # code enters the exception part only if trying wasn't successful.
        errorString = str(gooseError)  # regular error is not iterable, thus convert to string.
        if '404' in errorString:  # if error occurred, fill in Nones + raise the flag to True.
            gooseHeadLine = None
            gooseBody = None
            gooseErrorFlag = True

    if gooseErrorFlag == False:  # if no errors, fill in the parameters.
        gooseHeadLine = gooseArticle.title
        gooseBody = gooseArticle.cleaned_text

    # fill in our df with one of the libraries parameters, newspaper is chosen here arbitrary.
    singleInputDF.loc[0, 'HeadLine'] = newspaperHeadLine
    singleInputDF.loc[0, 'Body'] = newspaperBody

    # following 2 IFs: if one of the parameters is empty or None, replace it with the relevant data from the second
    # library, as it might have better content.
    if ((singleInputDF.loc[0, 'HeadLine'] == None) or (singleInputDF.loc[0, 'HeadLine'] == '')):
        singleInputDF.loc[0, 'HeadLine'] = gooseHeadLine

    if ((singleInputDF.loc[0, 'Body'] == None) or (singleInputDF.loc[0, 'Body'] == '')):
        singleInputDF.loc[0, 'Body'] = gooseBody

    # check if input is valid: if any body AND header extracted at all
    if ((singleInputDF.loc[0, 'HeadLine'] == None) or (singleInputDF.loc[0, 'HeadLine'] == '')):
        return 'input error'
    if ((singleInputDF.loc[0, 'Body'] == None) or (singleInputDF.loc[0, 'Body'] == '')):
        return 'input error'

    return singleInputDF



# myUrl = "http://www.bbc.com/news/world-us-canada-41419190"
myUrl = 'http://www.dofan.co.il/'
myDF = convertUrlToDF(myUrl)
print(myDF)
