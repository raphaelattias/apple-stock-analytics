# Useful starting lines
import numpy as np
from util.dataloader import *
from util.plots import *
from util.quotebankexploration import *
from util.wikipedia import *
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import seaborn as sns
import math

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
import numpy as np


#Vader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def predict_sentiment(quotes):
    new_quotes = quotes.copy()
    new_quotes.rename({'quotation': 'Quotation'}, axis = 1, inplace=True)
    
    analyzer = SentimentIntensityAnalyzer()
    # determine the sentiment of a quote in a corpus (positive, negative or neutral)
    def sentiment(quote) : 
        vs = analyzer.polarity_scores(quote)['compound']
        if (vs >=0.05) :
            return('positive')
        if (vs <= - 0.05) :
            return('negative') 
        else : return('neutral')  

    # determine the sentiment of a quote in a corpus (+1, -1 or 0)
    def sentiment_binary(quote) : 
        vs = analyzer.polarity_scores(quote)['compound']
        if (vs >=0.05) :
            return(1)
        if (vs <= - 0.05) :
            return(-1) 
        else : return(0) 

    new_quotes['sentiment'] = new_quotes['Quotation'].apply(sentiment) 

    return new_quotes

def correlation_stock_sentiment(quotes,stock):
    # separate the positive and negative quotes
    pos_quotes = quotes[quotes['sentiment'] == 'positive']
    neg_quotes = quotes[quotes['sentiment'] == 'negative']
    neut_quotes = quotes[quotes['sentiment'] == 'neutral']

    # plot the distribution of the neutral quotes according to time
    neut_per_day = pd.DataFrame(neut_quotes.groupby(neut_quotes.date.dt.date).count()['sentiment'])
    neut_per_day.index.rename('Date')
    neut_per_day.reset_index(inplace=True)

    # plot the distribution of the positive quotes according to time
    pos_per_day = pd.DataFrame(pos_quotes.groupby(pos_quotes.date.dt.date).count()['sentiment'])
    pos_per_day.index.rename('Date')
    pos_per_day.reset_index(inplace=True)

    # plot the distribution of the negative quotes according to time
    neg_per_day = pd.DataFrame(neg_quotes.groupby(neg_quotes.date.dt.date).count()['sentiment'])
    neg_per_day.index.rename('Date')
    neg_per_day.reset_index(inplace=True)

    ############
    # correlation 
    pos_per_day.rename({'date': 'Date'}, axis=1, inplace=True)
    pos_per_day['Date']= pd.to_datetime(pos_per_day['Date'], errors='coerce')
    stock_to_keep = stock[stock.Date.isin(set(stock.Date).intersection(set(pos_per_day.Date)))]
    pos_per_day_to_keep = pos_per_day[pos_per_day.Date.isin(set(stock.Date).intersection(set(pos_per_day.Date)))]
    print("Pearson pos", pearsonr(stock_to_keep.Liquidity,pos_per_day_to_keep.sentiment))
    
    neg_per_day.rename({'date': 'Date'}, axis=1, inplace=True)
    neg_per_day['Date']= pd.to_datetime(neg_per_day['Date'], errors='coerce')
    stock_to_keep = stock[stock.Date.isin(set(stock.Date).intersection(set(neg_per_day.Date)))]
    neg_per_day_to_keep = neg_per_day[neg_per_day.Date.isin(set(stock.Date).intersection(set(neg_per_day.Date)))]
    print("Pearson neg", pearsonr(stock_to_keep.Liquidity,neg_per_day_to_keep.sentiment))

    neut_per_day.rename({'date': 'Date'}, axis=1, inplace=True)
    neut_per_day['Date']= pd.to_datetime(neut_per_day['Date'], errors='coerce')
    stock_to_keep = stock[stock.Date.isin(set(stock.Date).intersection(set(neut_per_day.Date)))]
    neut_per_day_to_keep = neut_per_day[neut_per_day.Date.isin(set(stock.Date).intersection(set(neut_per_day.Date)))]
    print("Pearson neut", pearsonr(stock_to_keep.Liquidity,neut_per_day_to_keep.sentiment))

def fig_all_sentiments(quotes,stock):
    stock_name = "AAPL"
    # separate the positive and negative quotes
    pos_quotes = quotes[quotes['sentiment'] == 'positive']
    neg_quotes = quotes[quotes['sentiment'] == 'negative']
    neut_quotes = quotes[quotes['sentiment'] == 'neutral']

    # plot the distribution of the neutral quotes according to time
    neut_per_day = pd.DataFrame(neut_quotes.groupby(neut_quotes.date.dt.date).count()['sentiment'])
    neut_per_day.index.rename('Date')
    neut_per_day.reset_index(inplace=True)
    neut_per_day.rename({'date': 'Date'}, axis=1, inplace=True)
    neut_per_day['Date']= pd.to_datetime(neut_per_day['Date'], errors='coerce')
    
    # plot the distribution of the positive quotes according to time
    pos_per_day = pd.DataFrame(pos_quotes.groupby(pos_quotes.date.dt.date).count()['sentiment'])
    pos_per_day.index.rename('Date')
    pos_per_day.reset_index(inplace=True)
    pos_per_day.rename({'date': 'Date'}, axis=1, inplace=True)
    pos_per_day['Date']= pd.to_datetime(pos_per_day['Date'], errors='coerce')

    # plot the distribution of the negative quotes according to time
    neg_per_day = pd.DataFrame(neg_quotes.groupby(neg_quotes.date.dt.date).count()['sentiment'])
    neg_per_day.index.rename('Date')
    neg_per_day.reset_index(inplace=True)
    neg_per_day.rename({'date': 'Date'}, axis=1, inplace=True)
    neg_per_day['Date']= pd.to_datetime(neg_per_day['Date'], errors='coerce')



    all_quotes_per_day = pos_per_day.merge(neg_per_day,how="right",on="Date").merge(neut_per_day,how="left",on="Date")
    all_quotes_per_day.Date = all_quotes_per_day.Date.apply(lambda x : str(x))
    all_quotes_per_day.rename({"sentiment_x": "Positive", "sentiment_y": "Negative", "sentiment":"Neutral"},axis=1,inplace=True)
    all_quotes_per_day['All'] = all_quotes_per_day.Positive + all_quotes_per_day.Negative + all_quotes_per_day.Neutral

    all_quotes_per_day['All'] = all_quotes_per_day['All'].rolling(7, min_periods=1).mean().to_frame()
    all_quotes_per_day['Positive'] = all_quotes_per_day['Positive'].rolling(7, min_periods=1).mean().to_frame()
    all_quotes_per_day['Negative'] = all_quotes_per_day['Negative'].rolling(7, min_periods=1).mean().to_frame()
    all_quotes_per_day['Neutral'] = all_quotes_per_day['Neutral'].rolling(7, min_periods=1).mean().to_frame()

    fig = go.Figure()
    ymax = all_quotes_per_day.All.max()

    # Add Traces

    fig.add_trace(
        go.Bar(x=all_quotes_per_day.Date,
                y=all_quotes_per_day.All,
                name="All",
                marker_color='rgb(50,50,50)', opacity = 0.5,))
    fig['data'][0]['showlegend'] = True
    fig['data'][0]['name']='All'
    fig.add_trace(
        go.Bar(x=all_quotes_per_day.Date,
                y=all_quotes_per_day.Negative,
                name="Negative",
                visible=False,
                marker_color='rgb(165,37,30)'))

    fig.add_trace(
        go.Bar(x=all_quotes_per_day.Date,
                y=all_quotes_per_day.Positive,
                name="Positive",
                visible = False,
                marker_color='rgb(50,120,70)'))

    fig.add_trace(
        go.Bar(x=all_quotes_per_day.Date,
                y=all_quotes_per_day.Neutral,
                name="Neutral",
                visible=False,
                marker_color='rgb(50,90,200)'))

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=1,
                y=1.15,
                buttons=list([
                    dict(label="All quotes",
                        method="update",
                        args=[{"visible": [True, False, False, False]},
                            {"title": "All"}]),
                    dict(label="Negative",
                        method="update",
                        args=[{"visible": [True, True, False, False]},
                            {"title": "Negative quotes"}]),
                    dict(label="Positive",
                        method="update",
                        args=[{"visible": [True, False, True, False]},
                            {"title": "Positive quotes"}]),
                    dict(label="Neutral",
                        method="update",
                        args=[{"visible": [True, False, False, True]},
                            {"title": "Neutral quotes"}]),
                ]),
            )
        ])

    # Set title
    fig.update_layout(
        title_text="All quotes",
        xaxis_domain=[0.05, 1.0],
        yaxis_range =[0,ymax],
        xaxis_title_text='Date', # xaxis label
        yaxis_title_text='Frequency of quotes', # yaxis label
        bargap=0.1,
        bargroupgap = 0,
        barmode = "overlay",
        template = 'ggplot2'
    )
    fig.update_traces(marker_line_width = 0,
                    selector=dict(type="bar"))



    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.show()
    fig.write_html("figures/all_quotes_sentiment.html")


    all_quotes_per_day.Date = all_quotes_per_day.Date.apply(lambda x : str(x))
    all_quotes_per_day.rename({"sentiment_x": "Positive", "sentiment_y": "Negative", "sentiment":"Neutral"},axis=1,inplace=True)

    #all_quotes_per_day.Positive=(all_quotes_per_day.Positive-all_quotes_per_day.Positive.mean())/all_quotes_per_day.Positive.std()
    #all_quotes_per_day.Negative=(all_quotes_per_day.Negative-all_quotes_per_day.Negative.mean())/all_quotes_per_day.Negative.std()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ymax = max(all_quotes_per_day.Positive.max(), all_quotes_per_day.Negative.max())

    # Add Traces

    fig.add_trace(
        go.Bar(x=all_quotes_per_day.Date,
                y=all_quotes_per_day.Negative,
                name="Negative",
                    visible=True,
                marker_color='rgb(165,37,30)'))

    fig.add_trace(
        go.Bar(x=all_quotes_per_day.Date,
                y=all_quotes_per_day.Positive,
                name="Positive",
                visible = False,
                marker_color='rgb(50,120,70)'))

    fig.add_trace(go.Scatter(x=stock['Date'], 
                            y=(stock['Open']).interpolate(method="polynomial",
                            order=5), name = f"{stock_name} stock price", 
                            visible=True, 
                            marker_color='rgb(30,50,155)'),
                            secondary_y=True)


    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=1,
                y=1.15,
                buttons=list([
                    dict(label="Negative",
                        method="update",
                        args=[{"visible": [ True, False, True]},
                            {"title": "Negative quotes"}]),
                    dict(label="Positive",
                        method="update",
                        args=[{"visible": [False, True, True]},
                            {"title": "Positive quotes"}])
                ]),
            )
        ])

    # Set title
    fig.update_layout(
        title_text="Negative Quotes",
        xaxis_domain=[0.05, 1.0],
        yaxis_range =[0,ymax],
        xaxis_title_text='Date', # xaxis label
        yaxis_title_text='Frequency of quotes', # yaxis label
        bargap=0.1,
        bargroupgap = 0,
            template = 'ggplot2'

    )

    fig.update_yaxes( secondary_y=False, range=[0,1100])
    fig.update_yaxes( secondary_y=True, range=[0,90])

    fig.update_traces(marker_line_width = 0,  selector=dict(type="bar"))

    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
    fig.show()
    fig.write_html("figures/neg_pos_market.html")