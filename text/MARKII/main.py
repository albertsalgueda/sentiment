from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re

# INSTANTIATE OUR MODEL
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

#ENCODE AND CALCULATE SENTIMENT

sentence = 'I hate this shit'
tokens = tokenizer.encode(f'{sentence}', return_tensors = 'pt') 
#this returns a list of lists made of tokens, tokens[0] to grab the internal list

#DECODE
#tokenizer.decode(tokens[0])

#PASS THE TOKENS TO OUR MODEL
result = model(tokens)

"""
result:
SequenceClassifierOutput(loss=None, logits=tensor([[ 3.9135,  0.8185, -1.0532, -2.1354, -1.0108]],
       grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)

result.logits:
tensor([[ 3.9135,  0.8185, -1.0532, -2.1354, -1.0108]], grad_fn=<AddmmBackward0>)
"""
#COMPUTE THE SENTIMENT SCORING
sentiment = int(torch.argmax(result.logits)+1)

#TRY OUR CODE WITH YELP.COM REVIEWS
r = requests.get('https://www.yelp.com/biz/social-brew-cafe-pyrmont')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class':regex}) #everything with a class that matches a reges == all comments
reviews = [result.text for result in results] #we grab only the text from the class
#print(reviews)
#print(len(reviews))

#LOAD REVIEWS INTO DATAFRAME AND SCORE
import numpy as np
import pandas as pd

df = pd.DataFrame(np.array(reviews),columns=['review'])
#get one element of the data set
#df['review'].iloc[0] 

def sentiment_score(sentence):
    tokens = tokenizer.encode(f'{sentence}', return_tensors = 'pt') 
    result = model(tokens)
    sentiment = int(torch.argmax(result.logits)+1)
    return sentiment

#print(sentiment_score(df['review'].iloc[0]))
df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512])) 
#the [:512] is because the NPL model is limited to 512 tokens at a time
print(df)

def get_average(df):
    total = 0.0
    for raw in range(len(df)):
        total += df['sentiment'].iloc[raw] 
    average = float(total/len(df))
    return average

print(f'Your average sentiment score is {get_average(df)}')