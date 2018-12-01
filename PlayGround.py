import newspaper
import goose3
import pandas as pd
import nltk.data
import re
from nltk.corpus import brown
import nltk
from copy import deepcopy
import requests
import urllib
import httplib2
import urllib3


inputURL = 'https://www.amazon.com/'

# check if URL exists
try:
    request = requests.get(inputURL)
    if request.status_code != 200:  # url does not exist
        print('input error')
    else:  # url exists
        print('website exists')
except Exception as inputError:
    print('exception: input error')








'''try:
    h = httplib2.Http()
    resp = h.request(inputURL, 'HEAD')
    if (int(resp[0]['status']) < 400):
        print("website exists")
    else:
        print("error: NOT exists")'''

