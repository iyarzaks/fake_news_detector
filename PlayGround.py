import newspaper
import goose3
import pandas as pd
import nltk.data
import re
from nltk.corpus import brown
import nltk

my_dict = {'x':500, 'y':5874, 'z': 560}

key_max = max(my_dict.keys(), key=(lambda k: my_dict[k]))
key_min = min(my_dict.keys(), key=(lambda k: my_dict[k]))

print(key_max, ': ', my_dict[key_max])
print('Minimum Value: ',my_dict[key_min])



