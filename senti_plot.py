import sys
if sys.version_info[0] < 3:
    import got
else:
    import GetOldTweets3 as got

import pandas as pd
import warnings
import re
import numpy as np
import matplotlib.pyplot as plt

# NTLK functions
import nltk


from nltk.corpus import stopwords
from nltk import tokenize as tok
from nltk.stem.snowball import SnowballStemmer # load nltk's SnowballStemmer as variabled 'stemmer'

import string
from nltk.tag import StanfordNERTagger

df = pd.read_csv('./csvf/senti_quarantine20/tweets_topics_sentiment.csv')
df = df[['date','Sentiment_Score']]
dates = df['date'].to_numpy(dtype=str)
new_date = []
for date in dates:
    print(date)
    date = date[0:10]
    new_date.append(date)
new_date = new_date[::-1]

print(len(new_date))

scores = df['Sentiment_Score'].to_numpy(dtype=float)
scores = scores[::-1]

print(len(scores))

pin = 0
flag = 0
mean_l = []
for i in range(len(new_date)):
    if flag == 1:
        break
    i = pin
    record = new_date[i]
    sum = scores[i]
    for j in range(len(new_date)):
        pin = i + j + 1
        if pin >= len(new_date):
            flag = 1
            break
        if new_date[pin] == record:
            sum = sum + scores[pin]
        else:
            mean = sum/j
            print(mean)
            mean_l.append(mean)
            break
print(mean_l)
print(len(mean_l))

plt.plot(mean_l)
plt.savefig('./quarantine20.png')