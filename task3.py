import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np
#Vader
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def task3(quotes):
    
    analyzer = SentimentIntensityAnalyzer()
    # determine the sentiment of a quote in a corpus (positive, negative or neutral)
    def sentiment(quote) : 
        vs = analyzer.polarity_scores(quote)['compound']
        if (vs >=0.05) :
            return('positive')
        if (vs <= - 0.05) :
            return('negative') 
        else : return('neutral')  
    
    quotes['sentiment'] = quotes['quotation'].apply(sentiment) 
    
    # plot the distribution of the sentiments in the corpus 
    df_sent = quotes.groupby(['sentiment']).sum().reset_index()
    sns.barplot(data=df_sent, x='sentiment', y='numOccurrences')
    plt.ylabel('Frequency'); plt.xlabel('Sentiment')
    plt.title('Distribution of the quote sentiments')
    
    # separate the positive and negative quotes
    pos_quotes = quotes[quotes['sentiment'] == 'positive']
    neg_quotes = quotes[quotes['sentiment'] == 'negative']
    
    # plot the distribution of the positive quotes according to time
    pos_per_day = pos_quotes.groupby('date').count()['sentiment']
    pos_per_day
    fig, ax = plt.subplots(figsize=(12,8))
    pos_per_day.plot()
    plt.title('Distribution of the positive quotes according to time')
    plt.ylabel('Count')

    # plot the distribution of the negative quotes according to time
    neg_per_day = neg_quotes.groupby('date').count()['sentiment']
    neg_per_day
    fig, ax = plt.subplots(figsize=(12,8))
    neg_per_day.plot()
    plt.title('Distribution of the negative quotes according to time')
    plt.ylabel('Count')
    
    return None