import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import seaborn as sns
import math
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    
#analyzer = SentimentIntensityAnalyzer()


def sentiment_score(quote, analyzer) :
    """
    Determine the sentiment of a quote in a corpus.

    Args: 
        quote (string): corpus on which we want to determine the sentiment
        analyzer (SentimentIntensityAnalyzer()): from VADER Sentiment Analysis 
    Return:
        vs (float): the score which is normalized between -1(most extreme negative) and +1 (most extreme positive).
    """   
    vs =  analyzer.polarity_scores(quote)['compound'] 
    return vs  


def sentiment(quote, analyzer) : 
    """
    Determine the sentiment of a quote in a corpus according to the compound score.

    Args:
        quote (string): corpus on which we want to determine the sentiment
        analyser (SentimentIntensityAnalyzer()): from VADER Sentiment Analysis  
    Return:
        (string): the sentiment of the corpus (positive, negative or neutral).
    """    
    vs = sentiment_score(quote, analyzer)
    if (vs >=0.05) :
        return('positive')
    if (vs <= - 0.05) :
     return('negative') 
    else : return('neutral')  

# determine the sentiment of a quote in a corpus (+1, -1 or 0)
def sentiment_binary(quote, analyzer) : 
    """
    Determine the sentiment of a quote in a corpus according to the compound score.

    Args:
        quote (string): corpus on which we want to determine the sentiment
        analyser (SentimentIntensityAnalyzer()): from VADER Sentiment Analysis 
    Return: 
        (int): the sentiment of the corpus (+1 for positive, -1 for negative or 0 for neutral). 
    """    
    vs = sentiment(quote, analyzer)
    if (vs == 'positive') :
        return(1)
    if (vs == 'negative') :
        return(-1) 
    else : return(0)       
