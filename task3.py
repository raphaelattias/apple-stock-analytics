import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    fig = px.histogram(df_sent, x="sentiment", y='numOccurrences')
    fig.update_layout(
    title={
            'text' : 'Distribution of the quote sentiments',
            'x':0.5,
            'xanchor': 'center'},
    xaxis_title_text='Sentiments', # xaxis label
    yaxis_title_text='Frequency', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
    )
    fig.update_traces(opacity=0.75)
    fig.show()
    
    # separate the positive and negative quotes
    pos_quotes = quotes[quotes['sentiment'] == 'positive']
    neg_quotes = quotes[quotes['sentiment'] == 'negative']
    
    
    # plot the distribution of the positive quotes according to time
    pos_per_day = pd.DataFrame(pos_quotes.groupby(pos_quotes.date.dt.date).count()['sentiment'])
    pos_per_day.index.rename('Date')
    pos_per_day.reset_index(inplace=True)
    fig = px.histogram(pos_per_day, x=pos_per_day['date'], y=pos_per_day['sentiment'])
    fig.update_layout(
    title={
            'text' : 'Distribution of the positive quotes according to time',
            'x':0.5,
            'xanchor': 'center'},
    xaxis_title_text='date', # xaxis label
    yaxis_title_text='frequency of positive quotes', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
    )
    fig.update_traces(opacity=0.75)
    fig.show()
    

    # plot the distribution of the negative quotes according to time
    neg_per_day = pd.DataFrame(neg_quotes.groupby(neg_quotes.date.dt.date).count()['sentiment'])
    neg_per_day.index.rename('Date')
    neg_per_day.reset_index(inplace=True)
    fig = px.histogram(neg_per_day, x=neg_per_day['date'], y=neg_per_day['sentiment'])
    fig.update_layout(
    title={
            'text' : 'Distribution of the negative quotes according to time',
            'x':0.5,
            'xanchor': 'center'},
    xaxis_title_text='date', # xaxis label
    yaxis_title_text='frequency of negative quotes', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
    )
    fig.update_traces(opacity=0.75)
    fig.show()
    
    return None