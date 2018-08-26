import pandas as pd
from project_utils import *

data = pd.read_csv('Classified_Data_kaggle.csv', engine='python')
artcles_list = data['Body'].values.astype('U').tolist()
labels = data['Label (1=true)'].values.astype('U').tolist()
to_bag_of_words(artcles_list,labels)