import newspaper
import goose3
import pandas as pd
import nltk.data
import re
from nltk.corpus import brown
import nltk

mystr = 'hello i am alon'
print(type(mystr))

if isinstance(mystr, str):
    print('yes')
if not (isinstance(mystr, str)):
    print('no')

isinstance("this is a string", str)