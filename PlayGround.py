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
inputURL = 'https://portal.azure.com/#@iyarzaksgmail.onmicrosoft.com/resource/subscriptions/865382c9-0db9-4ff2-998a-403212b84069/resourceGroups/fake_news_project/providers/Microsoft.Sql/servers/iz/databases/fake_news_DB/overview'

# check if URL exists
'''try:
    request = requests.get(inputURL)
    if request.status_code != 200:  # url does not exist
        print('input error')
    else:  # url exists
        print('website exists')
except Exception as inputError:
    print('exception: input error')'''




from urllib.parse import urlparse
parsed_uri = urlparse(inputURL)
result = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
print(result)

url= inputURL
domain=url.split("//")[-1].split("/")[0]
print (domain)







'''try:
    h = httplib2.Http()
    resp = h.request(inputURL, 'HEAD')
    if (int(resp[0]['status']) < 400):
        print("website exists")
    else:
        print("error: NOT exists")'''

