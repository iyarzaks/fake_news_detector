import pandas as pd


data = pd.read_csv('Classified_Data_kaggle.csv', engine='python')
print (data['URLs'])