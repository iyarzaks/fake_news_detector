import newspaper
import goose3
import pandas as pd
import nltk.data
import re
from nltk.corpus import brown
import nltk
from copy import deepcopy

from urllib.parse import urlparse
data = urlparse("https://moodle.technion.ac.il/mod/forum/view.php?id=546044")
print(data.netloc)