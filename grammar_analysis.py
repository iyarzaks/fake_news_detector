import newspaper
import goose3
import pandas as pd
import nltk.data
import re
from nltk.corpus import brown
import nltk
import json
from copy import deepcopy


# reads input as a data csv file of classified articles. It analyzes grammar parameters based on the data and creates
# a new csv file with all the results for each article.
def data_grammar_analyze(csv_data_file_name):
    original_df = pd.read_csv(csv_data_file_name, encoding = "ISO-8859-1")  # ISO encoding is just a standard way to avoid some errors.

    analysis_df = pd.DataFrame()
    analysis_df['URLs'] = 0
    analysis_df['Author'] = []
    analysis_df['Date'] = 0
    analysis_df['HeadLine'] = 0
    analysis_df['Body'] = 0

    df_row_count = 0  # accessing index when filling data in original_df, it avoids errors (notice at the end of code when it's used)
    for index, row in original_df.iterrows():  # iterate the df
        gooseErrorFlag = False # flag stating if the goose article was successfully extracted.
        currURL = original_df.at[index, 'URLs']
        currURL = currURL.replace('\r\n', '') # remove the "\r\n" at the end of URL, it avoids errors.

        # filling newspaper parameters.
        newspaperArticle = newspaper.Article(currURL)
        newspaperArticle.download()
        if newspaperArticle.download_state == 2:  # checks validation of downloading: 2=downloaded, 1=unsuccessful download
            newspaperArticle.parse()
            newspaperAuthor = newspaperArticle.authors
            newspaperDate = newspaperArticle.publish_date
            newspaperHeadLine = newspaperArticle.title
            newspaperBody = newspaperArticle.text
        else: # if article couldn't download, fill None in all parameters.
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

        # fill in analysis_df with one of the libraries parameters, newspaper is chosen here arbitrary.
        analysis_df.loc[df_row_count, 'URLs'] = currURL
        analysis_df.loc[df_row_count, 'Author'] = str(newspaperAuthor) # df can't have list as a value, thus convert to string.
        analysis_df.loc[df_row_count, 'Date'] = newspaperDate
        analysis_df.loc[df_row_count, 'HeadLine'] = newspaperHeadLine
        analysis_df.loc[df_row_count, 'Body'] = newspaperBody

        # following 4 IFs: if one of the parameters is empty or None, replace it with the relevant data from the second
        # library, as it might have better content.
        if ( (analysis_df.loc[df_row_count, 'Author']==None) or (analysis_df.loc[df_row_count, 'Author']=='[]') ):
            analysis_df.loc[df_row_count, 'Author'] = str(gooseAuthor)

        if (analysis_df.loc[df_row_count, 'Date'] == None):
            analysis_df.loc[df_row_count, 'Date'] = gooseDate

        if ((analysis_df.loc[df_row_count, 'HeadLine'] == None) or (analysis_df.loc[df_row_count, 'HeadLine'] == '')):
            analysis_df.loc[df_row_count, 'HeadLine'] = gooseHeadLine

        if ((analysis_df.loc[df_row_count, 'Body'] == None) or (analysis_df.loc[df_row_count, 'Body'] == '')):
            analysis_df.loc[df_row_count, 'Body'] = gooseBody

        df_row_count = df_row_count + 1

    # ------after reading all articles, start analyzing------

    initDataPerArticle = {}
    for index, row in original_df.iterrows():
        articleAttrs = {}

        # check only articles with a header and a body
        headerFlagError = False
        bodyFlagError = False
        currentURL = original_df.at[index, 'URLs']
        currentLabel = original_df.at[index, 'Label (1=true)']
        if currentLabel!='0' and currentLabel!='1':
            currentLabel='missing label'
        currentHeadLineList = original_df.at[index, 'Headline']
        currentBodyList = original_df.at[index, 'Body']
        if (currentHeadLineList == []) or (not(isinstance(currentHeadLineList, str))):
            headerFlagError = True
        if (currentBodyList == []) or (not(isinstance(currentBodyList, str))):
            bodyFlagError = True

        if (headerFlagError == False) and (bodyFlagError == False):  # relate only to articles with a header and a body
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
            articleAttrs['label'] = currentLabel

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
            articleAttrs['label'] = 0

        initDataPerArticle[currentURL] = articleAttrs

    result_df = deepcopy(original_df)
    result_df['numOfHeaderSentences'] = 0
    result_df['numOfBodySentences'] = 0
    result_df['meanHeaderLen'] = 0
    result_df['minHeaderLength'] = 0
    result_df['maxHeaderLength'] = 0
    result_df['meanBodyLen'] = 0
    result_df['minBodyLength'] = 0
    result_df['maxBodyLength'] = 0
    result_df['headerMisspellRate'] = 0
    result_df['bodyMisspellRate'] = 0
    result_df['headerGrammarMaxKey'] = 0
    result_df['headerGrammarMaxTimes'] = 0
    result_df['headerGrammarMinKey'] = 0
    result_df['headerGrammarMinTimes'] = 0
    result_df['bodyGrammarMaxKey'] = 0
    result_df['bodyGrammarMaxTimes'] = 0
    result_df['bodyGrammarMinKey'] = 0
    result_df['bodyGrammarMinTimes'] = 0

    result_df_row_count = 0
    for url, attrs in initDataPerArticle.items():
        result_df.loc[result_df_row_count, 'numOfHeaderSentences'] = attrs['numOfHeaderSentences']
        result_df.loc[result_df_row_count, 'numOfBodySentences'] = attrs['numOfBodySentences']
        result_df.loc[result_df_row_count, 'meanHeaderLen'] = attrs['meanHeaderLen']
        result_df.loc[result_df_row_count, 'minHeaderLength'] = attrs['minHeaderLength']
        result_df.loc[result_df_row_count, 'maxHeaderLength'] = attrs['maxHeaderLength']
        result_df.loc[result_df_row_count, 'meanBodyLen'] = attrs['meanBodyLen']
        result_df.loc[result_df_row_count, 'minBodyLength'] = attrs['minBodyLength']
        result_df.loc[result_df_row_count, 'maxBodyLength'] = attrs['maxBodyLength']
        result_df.loc[result_df_row_count, 'headerMisspellRate'] = attrs['headerMisspellRate']
        result_df.loc[result_df_row_count, 'bodyMisspellRate'] = attrs['bodyMisspellRate']
        result_df.loc[result_df_row_count, 'headerGrammarMaxKey'] = attrs['headerGrammarMaxKey']
        result_df.loc[result_df_row_count, 'headerGrammarMaxTimes'] = attrs['headerGrammarMaxTimes']
        result_df.loc[result_df_row_count, 'headerGrammarMinKey'] = attrs['headerGrammarMinKey']
        result_df.loc[result_df_row_count, 'headerGrammarMinTimes'] = attrs['headerGrammarMinTimes']
        result_df.loc[result_df_row_count, 'bodyGrammarMaxKey'] = attrs['bodyGrammarMaxKey']
        result_df.loc[result_df_row_count, 'bodyGrammarMaxTimes'] = attrs['bodyGrammarMaxTimes']
        result_df.loc[result_df_row_count, 'bodyGrammarMinKey'] = attrs['bodyGrammarMinKey']
        result_df.loc[result_df_row_count, 'bodyGrammarMinTimes'] = attrs['bodyGrammarMinTimes']

        result_df_row_count = result_df_row_count + 1

    result_df = result_df.drop('Unnamed: 4', 1)
    result_df = result_df.drop('Unnamed: 5', 1)
    result_df.to_csv('data_grammar_analysis.csv', encoding='utf-8', index=False)