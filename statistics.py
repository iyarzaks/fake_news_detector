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
import tokenize
import heapq


# reads all data from a csv file and analyzes raw data. Creates a csv file with most 'n' appearing domains,
# and returns a dict of the other raw data such as num of fake or true articles, mean len of an article etc.

def show_data_statistics(all_data):

    domain_dict = {}
    fake_number = 0
    true_number = 0
    article_len_dict = {}

    for index, row in all_data.iterrows():
        url = row['URLs']
        header = row['Headline']
        body = row['Body']
        label = row['Label (1=true)']
        myString = 'alon is king'  # easter egg :)

        # skip rows without all data
        if ( (type(url) != type(myString)) or (type(header) != type(myString)) or (type(body) != type(myString))
            or (type(label) != type(myString)) ):
            continue

        domain = url.split("//")[-1].split("/")[0]
        if domain in domain_dict.keys():
            domain_dict[domain] = domain_dict[domain] + 1
        else:
            domain_dict[domain] = 1

        if label == '1':
            true_number = true_number + 1
        else:
            fake_number = fake_number + 1

        # get all words of an article
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        BodySentences = tokenizer.tokenize(body)
        sumToAdd = 0
        for bodySentence in BodySentences:
            words = re.sub("[^\w]", " ", bodySentence).split()
            sumToAdd = sumToAdd + len(words)

        article_len_dict[index + 1] = sumToAdd


    # compute mean lean of article
    count = 0
    sum_of_words = 0
    for key in article_len_dict:
        count += 1
        sum_of_words += article_len_dict[key]

    mean_article_len = sum_of_words / count

    # create csv of top 8 domains to easily create a graph later
    result_df = pd.DataFrame(columns=['domain_name', 'num_of_instances'])

    num_of_domains_to_show = 8
    d = domain_dict
    maximums_dict = {k: d[k] for k in heapq.nlargest(num_of_domains_to_show, d, key=lambda k: d[k])}
    count = 0
    for domain, instance_num in maximums_dict.items():
        result_df.loc[count] = [domain, instance_num]
        count+=1

    result_df.to_csv('statistics.csv', encoding='utf-8', index=False)


    # create dict of results to return
    total_articles = fake_number + true_number
    resultDict = {}
    resultDict['total_articles'] = total_articles
    resultDict['true_articles'] = true_number
    resultDict['fake_articles'] = fake_number
    resultDict['mean_article_len'] = mean_article_len
    return resultDict