import newspaper
import goose3
import pandas as pd
import nltk.data
import re
from nltk.corpus import brown
import nltk
import json
from copy import deepcopy
import requests



def convertUrlToDF(inputURL):
    singleInputDict = {}
    singleInputDict['URLs'] = inputURL

    gooseErrorFlag = False  # flag stating if the goose article was successfully extracted.

    #check if URL exists
    try:
        request = requests.get(inputURL)
        if request.status_code != 200: # url does not exist
            return 'input error'
        else: # url exists
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

            singleInputDict['HeadLine'] = newspaperHeadLine
            singleInputDict['Body'] = newspaperBody
            # following 2 IFs: if one of the parameters is empty or None, replace it with the relevant data from the second
            # library, as it might have better content.
            if ( (singleInputDict['HeadLine'] == None) or (singleInputDict['HeadLine'] == '') or (singleInputDict['HeadLine'] == 0) ):
                singleInputDict['HeadLine'] = gooseHeadLine

            if ((singleInputDict['Body'] == None) or (singleInputDict['Body'] == '')  or (singleInputDict['Body'] == 0) ):
                singleInputDict['Body'] = gooseBody

            # check if input is valid: if any body AND header extracted at all
            if ((singleInputDict['HeadLine'] == None) or (singleInputDict['HeadLine'] == '')  or (singleInputDict['HeadLine'] == 0) ):
                return 'input error'
            if ((singleInputDict['Body'] == None) or (singleInputDict['Body'] == '')  or (singleInputDict['Body'] == 0) ):
                return 'input error'
            df = pd.DataFrame()
            df.loc[0, 'URLs'] = singleInputDict['URLs']
            df.loc[0, 'HeadLine'] = singleInputDict['HeadLine']
            df.loc[0, 'Body'] = singleInputDict['Body']
            return df


    except Exception as inputError: # trying request for url wasn't successful - input error
        return 'input error'
