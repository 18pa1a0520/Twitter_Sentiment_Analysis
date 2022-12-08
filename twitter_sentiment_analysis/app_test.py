import os
from flask import Flask, render_template, request
from flask import send_from_directory
import tweepy
from textblob import TextBlob
import numpy as np
import pandas as pd
import re 
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
#nltk.download('punkt')   
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tweepy import OAuthHandler 
import seaborn as sns
import itertools
import collections

app = Flask(__name__)
text1=""

hashtags_file_path = 'uploads/hashtags_searched.txt'

#define variables for tweepy
consumer_key = 'i9QvfW3wQEeyxRzLO2opUJQZP' #enter consumer key
consumer_secret = 'CPB6DkrdUm5RjN6LOphNRwn0zU0gKqujh621zfDDPvrxVuRRdB' #enter consumer key secret
access_token = '1514838198706667525-aRiziFQW743BlYyBG5nOb64Tv8iJon' #enter access token
access_token_secret = 'LxsTDEv6lAtSihBfcCjGQlxlL3STnGOf7ZosKBrV0mr3u' #enter access token secret

def remove_url(txt):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

def graph_plot(tweets):
    global text1
    tweets_no_urls = [remove_url(tweet.text) for tweet in tweets]
    sentiment_objects = [TextBlob(tweet) for tweet in tweets_no_urls]
    sentiment_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_objects]
    sentiment_df = pd.DataFrame(sentiment_values, columns=["polarity", "tweet"])
    print(sentiment_df.head())
    #fig, ax = plt.subplots(figsize=(8, 6))
    #sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1], ax=ax, color="blue")
    ax = sentiment_df.plot.hist(bins=8)#[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1])
    for rect in ax.patches:
      if rect.get_x() > 0:
        rect.set_color('green')
      elif rect.get_x()==0:
        rect.set_color('blue')
      else:
        rect.set_color('red')
    plt.title("Sentiments from Tweets on " + text1)
    plt.show()
    sentiment_df = sentiment_df[sentiment_df.polarity != 0]
    #fig, ax = plt.subplots(figsize=(8, 6))
    
    ax = sentiment_df.plot.hist(bins=8)#[-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1])#, ax=ax, color="blue")
    for rect in ax.patches:
      if rect.get_x() > 0:
        rect.set_color('green')
      else:
        rect.set_color('red')
    plt.title("Sentiments from Tweets on "+text1)
    plt.show()

def cleanText(text):
  text = text.lower()
  # Removes a, text)
    
  # Removes anyll mentions (@username) from the tweet since it is of no use to us
  text = re.sub(r'(@[A-Za-z0-9_]+)', '', text)

  #Removes all link in text
  text = re.sub('http://\S+|https://\S+', '', text)

  # Only considers the part of the string with char between a to z or digits and whitespace characters
  # Basically removes punctuation
  text = re.sub(r'[^\w\s]', '', text)

  # Removes stop words that have no use in sentiment analysis 
  text_tokens = word_tokenize(text)
  text = [word for word in text_tokens if not word in stopwords.words()]

  text = ' '.join(text)
  return text

def stem(text):
  # This function is used to stem the given sentence
  porter = PorterStemmer()
  token_words = word_tokenize(text)
  stem_sentence = []
  for word in token_words:
    stem_sentence.append(porter.stem(word))
  return " ".join(stem_sentence)

def sentiment(cleaned_text):
  # Returns the sentiment based on the polarity of the input TextBlob object
  if cleaned_text.sentiment.polarity > 0:
    return 'positive'
  elif cleaned_text.sentiment.polarity < 0:
    return 'negative'
  else:
    return 'neutral'
    
#setup twitter authentication
auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)

# home page
@app.route('/')
def home():
   return render_template('index.html')


@app.route('/analyze', methods=['POST','GET'])
def analyzeTweet():
    global text1
    if request.method == 'GET':
        return render_template('index.html')
    else:
        hashtag = request.form['hashtag']
        text1 = hashtag
        f = open(hashtags_file_path,'a')
        if hashtag:
            tweets = []
            polaritylist = []
            subjectivitylist = []
            sentiments = []
            #tweets = api.search(hashtag, rpp=100, since_id=1, count=5000)
            fetched_data = api.search(hashtag, count = 5000)
            graph_plot(fetched_data)
            for tweet in fetched_data:
                txt = tweet.text
                clean_txt = cleanText(txt) # Cleans the tweet
                stem_txt = TextBlob(stem(clean_txt)) # Stems the tweet
                sent = sentiment(stem_txt) # Gets the sentiment from the tweet
                subjectivitylist.append(TextBlob(txt).subjectivity)
                tweets.append((txt, clean_txt, sent))
            if len(tweets)==0:
                return render_template('analyze.html', error_message='No tweets found!')
            if len(subjectivitylist) > 0:
                subjectivity = round(np.mean(subjectivitylist),3)*100
            print("subjectivity:",subjectivity)
            # Converting the list into a pandas Dataframe
            df = pd.DataFrame(tweets, columns= ['tweets', 'clean_tweets','sentiment'])

            # Dropping the duplicate values just in case there are some tweets that are copied and then stores the data in a csv file
            df = df.drop_duplicates(subset='clean_tweets')
            df.to_csv('data.csv', index= False)
            ptweets = df[df['sentiment'] == 'positive']
            p_perc = 100 * len(ptweets)/len(tweets)
            ntweets = df[df['sentiment'] == 'negative']
            n_perc = 100 * len(ntweets)/len(tweets)
            neutral = 100 - p_perc - n_perc
            print(f'Positive tweets {p_perc} %')
            print(f'Neutral tweets {neutral} %')
            print(f'Negative tweets {n_perc} %')
            sentiments = [p_perc, n_perc, neutral, subjectivity]
            sentiments = [ '%.3f' % elem for elem in sentiments ]
            f.write(hashtag + ' ' + '=> Positive: ' + str(p_perc) +'\n' + '=> Negative: ' + str(n_perc) + '\n'+'=> Neutral: '+str(neutral))
            f.close()
            return render_template('analyze.html', sentiments = sentiments, hashtag=hashtag)
           
        else:
            return render_template('analyze.html', error_message="Enter some hashtag!")



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False,threaded=False)
