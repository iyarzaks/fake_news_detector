import newspaper
import goose3
import pandas as pd
import nltk.data
import re
from nltk.corpus import brown
import nltk
from copy import deepcopy

originalDataFrame = pd.read_csv('Classified_Data_kaggle.csv', encoding = "ISO-8859-1")
originalDataFrame = originalDataFrame.iloc[0:2]
newOrderDF = deepcopy(originalDataFrame)
newOrderDF['numOfHeaderSentences'] = 'hey'
x=0