import sys
if sys.version_info[0] < 3:
    import got
else:
    import GetOldTweets3 as got

import pandas as pd
import warnings
import re
import numpy as np

# NTLK functions
import nltk
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk import tokenize as tok
from nltk.stem.snowball import SnowballStemmer # load nltk's SnowballStemmer as variabled 'stemmer'

import string
from nltk.tag import StanfordNERTagger
search_terms = ['coronavirus']


## scrape data from twitter for the above search terms
tweet_df_all = pd.DataFrame()
for term in search_terms:
    print(term)
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(term)\
                                               .setSince("2020-04-11")\
                                               .setUntil("2020-04-15")\
                                               .setNear("NYC").setMaxTweets(5000)
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)
    tweet_list = [[tweet[x].id,
                  tweet[x].author_id,
                  tweet[x].text,
                  tweet[x].retweets,
                  tweet[x].permalink,
                  tweet[x].date,
                  tweet[x].formatted_date,
                  tweet[x].favorites,
                  tweet[x].mentions,
                  tweet[x].hashtags,
                  tweet[x].geo,
                  tweet[x].urls
                 ]for x in range(0, len(tweet))]
    tweet_df = pd.DataFrame(tweet_list)
    tweet_df['search_term'] = term
    tweet_df_all = tweet_df_all.append(tweet_df)

tweet_df_all.columns = ['id','author_id','text','retweets','permalink','date','formatted_date','favorites','mentions','hashtags','geo','urls','search_term']
tweet_df_all.to_csv('./all_tweets.csv', index=False)


warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA
# Tf-Idf and Clustering packages
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tweet_df_comp = pd.read_csv('./all_tweets.csv')

isURL = re.compile(r'http[s]?:// (?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
                   re.VERBOSE | re.IGNORECASE)
isRTusername = re.compile(r'^RT+[\s]+(@[\w_]+:)', re.VERBOSE | re.IGNORECASE)  # r'^RT+[\s]+(@[\w_]+:)'
isEntity = re.compile(r'@[\w_]+', re.VERBOSE | re.IGNORECASE)


# Helper functions
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

    # Show top n keywords for each topic


def show_topics(vectorizer, lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


def clean_tweet(row):
    row = isURL.sub("", row)
    row = isRTusername.sub("", row)
    row = isEntity.sub("", row)
    return row


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in tok.sent_tokenize(text) for word in tok.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# %%

# remove urls and retweets and entities from the text
tweet_df_comp['text_clean'] = tweet_df_comp['text'].apply(lambda row: clean_tweet(row))

# remove punctuations
RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])
tweet_df_comp['text_clean'] = tweet_df_comp['text_clean'].str.replace(RE_PUNCTUATION, "")
tweet_df_comp.head()

# %%

# List of stopwords
stop_words = stopwords.words('english')  # import stopwords from NLTK package
readInStopwords = pd.read_csv("./pre_process/twitterStopWords.csv",
                              encoding='ISO-8859-1')  # import stopwords from CSV file as pandas data frame
readInStopwords = readInStopwords.wordList.tolist()  # convert pandas data frame to a list
readInStopwords.append('http')
readInStopwords.append('https')

# add in search terms as topic extraction is performed within each search topic,
# we do not want the word or valriation of the word captured as a topic word
search_terms_revised = ['coronavirus','corona', 'virus', 'covid','19','covid-19','people','nyc','us','new','york','de']
readInStopwords.extend(search_terms)
readInStopwords.extend(search_terms_revised)

stop_list = stop_words + readInStopwords  # combine two lists i.e. NLTK stop words and CSV stopwords
stop_list = list(set(stop_list))  # strore only unique values

# %%

# parameter for lda, i am selecrign 3 topic and 4 words for each of the search terms
number_topics = 7
number_words = 10

# %%

tweets_all_topics = pd.DataFrame()
# term frequency modelling
for terms in tweet_df_comp['search_term'].unique():
    print(terms)
    tweets_search_topics = tweet_df_comp[tweet_df_comp['search_term'] == terms].reset_index(drop=True)
    corpus = tweets_search_topics['text_clean'].tolist()
    # print(corpus)
    tf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.00, stop_words=stop_list,
                                    tokenizer=tokenize_only)  # Use tf (raw term count) features for LDA.
    tf = tf_vectorizer.fit_transform(corpus)

    # Create and fit the LDA model
    model = LDA(n_components=number_topics, n_jobs=-1)
    id_topic = model.fit(tf)
    # Print the topics found by the LDA model
    print("Topics found via LDA:")
    topic_keywords = show_topics(vectorizer=tf_vectorizer, lda_model=model, n_words=number_words)
    # Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
    df_topic_keywords = df_topic_keywords.reset_index()
    df_topic_keywords['topic_index'] = df_topic_keywords['index'].str.split(' ', n=1, expand=True)[[1]].astype('int')
    print(df_topic_keywords)

    ############ get the dominat topic for each document in a data frame ###############
    # Create Document — Topic Matrix
    lda_output = model.transform(tf)
    # column names
    topicnames = ["Topic" + str(i) for i in range(model.n_components)]
    # index names
    docnames = ["Doc" + str(i) for i in range(len(corpus))]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic
    df_document_topic = df_document_topic.reset_index()

    # combine all the search terms into one data frame
    tweets_topics = tweets_search_topics.merge(df_document_topic, left_index=True, right_index=True, how='left')
    tweets_topics_words = tweets_topics.merge(df_topic_keywords, how='left', left_on='dominant_topic',
                                              right_on='topic_index')
    tweets_all_topics = tweets_all_topics.append(tweets_topics_words)

# %%

tweets_all_topics = tweets_all_topics.reset_index(drop=True)
print(tweets_all_topics.shape)
tweets_all_topics.head()

# %%

tweets_all_topics.to_csv('./processed_data/tweets_all_topics.csv', index=False)

# Python program to generate WordCloud

# importing all necessery modules
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

# Reads 'Youtube04-Eminem.csv' file
stopwords = stop_list

comment_words = ''


# iterate through the csv file
for val in corpus:

    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens) + " "



wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(comment_words)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

#plt.show()
plt.savefig('./wc.png')