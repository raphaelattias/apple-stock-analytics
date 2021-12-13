import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np
import yfinance as yf
from dataloader import *
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
from plotly.subplots import make_subplots


from ipywidgets import interactive, HBox, VBox, Checkbox

#Vader
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def task3(quotes):
    stock_name = "AAPL"
    year = 2019
    year_start = 2015
    year_end = 2020
    # Find the days of high volatility 
    stock = yf.download(stock_name, start=f'{year_start}-01-01', end=f'{year_end}-12-31', progress = False)
    stock.reset_index(inplace=True)

    quotes = pd.concat([load_quotes(i, 'processed quotes') for i in range(year_start,year_end+1)])
    quotes.rename({'quotation': 'Quotation'}, axis = 1, inplace=True)
    
    analyzer = SentimentIntensityAnalyzer()
    # determine the sentiment of a quote in a corpus (positive, negative or neutral)
    def sentiment(quote) : 
        vs = analyzer.polarity_scores(quote)['compound']
        if (vs >=0.05) :
            return('positive')
        if (vs <= - 0.05) :
            return('negative') 
        else : return('neutral')  

    quotes['sentiment'] = quotes['Quotation'].apply(sentiment) 

    # plot the distribution of the sentiments in the corpus 
    df_sent = quotes.groupby(['sentiment']).sum().reset_index()

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

    fig = px.bar(pos_per_day, x=pos_per_day['date'], y=pos_per_day['sentiment'])
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
    fig.update_traces(marker_line_width = 0,
                    selector=dict(type="bar"))

    fig.update_layout(bargap=0.1,
                    bargroupgap = 0,
                    )
    fig.show()


    fig = px.bar(neg_per_day, x=neg_per_day['date'], y=neg_per_day['sentiment'])
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
    fig.update_traces(marker_line_width = 0,
                    selector=dict(type="bar"))

    fig.update_layout(bargap=0.1,
                    bargroupgap = 0,
                    )
    fig.show()
    fig.write_html("figures/pos_neg_neutral.html")

    ##########


# Initialize figure


    all_quotes_per_day = pos_per_day.merge(neg_per_day,how="right",on="date").merge(neut_per_day,how="left",on="date")
    all_quotes_per_day.Date = all_quotes_per_day.date.apply(lambda x : str(x))
    all_quotes_per_day.rename({"sentiment_x": "Positive", "sentiment_y": "Negative", "sentiment":"Neutral"},axis=1,inplace=True)
    all_quotes_per_day['All'] = all_quotes_per_day.Positive + all_quotes_per_day.Negative + all_quotes_per_day.Neutral


    fig = go.Figure()
    ymax = all_quotes_per_day.All.max()

    # Add Traces

    fig.add_trace(
        go.Bar(x=all_quotes_per_day.date,
                y=all_quotes_per_day.All,
                name="All",
                marker_color='rgb(30,50,130)', opacity = 0.5,))
    fig['data'][0]['showlegend'] = True
    fig['data'][0]['name']='All'
    fig.add_trace(
        go.Bar(x=all_quotes_per_day.date,
                y=all_quotes_per_day.Negative,
                name="Negative",
                visible=False,
                marker_color='rgb(165,37,30)'))

    fig.add_trace(
        go.Bar(x=all_quotes_per_day.date,
                y=all_quotes_per_day.Positive,
                name="Positive",
                visible = False,
                marker_color='rgb(50,120,70)'))

    fig.add_trace(
        go.Bar(x=all_quotes_per_day.date,
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
                    dict(label="All",
                        method="update",

                        args=[{"visible": [True, False, False, False]},
                            {"title": "All quotes"}]),
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
    ##########
# Initialize figure


    all_quotes_per_day.date = all_quotes_per_day.date.apply(lambda x : str(x))
    all_quotes_per_day.rename({"sentiment_x": "Positive", "sentiment_y": "Negative", "sentiment":"Neutral"},axis=1,inplace=True)

    #all_quotes_per_day.Positive=(all_quotes_per_day.Positive-all_quotes_per_day.Positive.mean())/all_quotes_per_day.Positive.std()
    #all_quotes_per_day.Negative=(all_quotes_per_day.Negative-all_quotes_per_day.Negative.mean())/all_quotes_per_day.Negative.std()

    ma = px.scatter(x=stock['Date'], y=stock['Open'],trendline="rolling", trendline_options=dict(window=25)).data[1]['y']

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ymax = max(all_quotes_per_day.Positive.max(),all_quotes_per_day.Negative.max())

    # Add Traces

    fig.add_trace(
        go.Bar(x=all_quotes_per_day.date,
                y=all_quotes_per_day.Negative,
                name="Negative",
                    visible=True,
                marker_color="red"))

    fig.add_trace(
        go.Bar(x=all_quotes_per_day.date,
                y=all_quotes_per_day.Positive,
                name="Positive",
                visible = False,
                marker_color="green"))

    fig.add_trace(go.Scatter(x=stock['Date'], y=(stock['Close']-stock['Open']).interpolate(method="polynomial",order=5), name = f"{stock_name} stock price", visible=True, marker_color='blue'),secondary_y=True)


    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.57,
                y=1.2,
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
        bargroupgap = 0
    )
    fig.update_traces(marker_line_width = 0,  selector=dict(type="bar"))
    fig.show()

    # Set title
    fig.update_layout(
        title_text="All quotes",
        xaxis_domain=[0.05, 1.0],
        yaxis_range =[0,ymax],
        xaxis_title_text='Date', # xaxis label
        yaxis_title_text='Frequency of quotes', # yaxis label
        bargap=0.1,
        bargroupgap = 0
    )
    fig.update_traces(marker_line_width = 0,  selector=dict(type="bar"))
    fig.show()
    
    return None