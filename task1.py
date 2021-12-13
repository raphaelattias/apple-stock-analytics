# Useful starting lines
import numpy as np
from util.dataloader import *
from util.plots import *
from util.finance import stock, compare
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

pio.renderers.default = "notebook_connected"

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

def load_stock(stock_name = "AAPL",year_start = 2008,year_end= 2020):
    stock = yf.download(stock_name, start=f'{year_start}-09-01', end=f'{year_end}-05-31', progress = False)
    stock.reset_index(inplace=True)
    stock['Liquidity'] = stock['Volume']*(stock['Close']+stock['Open'])/2

    return stock

def high_volatility(stock, quantile=0.98):
    stock['Yearly Percentile'] = stock.apply(lambda x: x['Liquidity'] > np.quantile(stock[stock.Date.dt.year == x.Date.year]['Liquidity'], q = quantile), axis=1)
    stock['Yearly Percentile'] = stock['Yearly Percentile'].apply(lambda x : f"Top {int(100-quantile*100)}%" if x else f"Lower {int(quantile*100)}%")

    return stock

def weekly_liquidity(stock, quantile=0.98):
    weekly = pd.DataFrame(stock.resample('W', on='Date')['Liquidity'].sum())
    weekly.index.rename('Date')
    weekly.reset_index(inplace=True)
    weekly = high_volatility(weekly, quantile)
    pio.renderers.default = "notebook_connected"

    year_start = stock.Date.dt.year.min()
    year_end = stock.Date.dt.year.max()
    fig = px.bar(weekly, x='Date', y='Liquidity', color='Yearly Percentile', title=f"Liquidity traded for the $AAPL stock between {year_start} and {year_end}",     \
                    color_discrete_map={
                    f"Top {int(100-quantile*100)}%": 'rgb(180,37,30)',
                    f"Lower {int(quantile*100)}%": 'rgb(30,50,155)'
                })
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
    fig.update_traces(marker_line_width = 0,
                        selector=dict(type="bar"))

    fig.update_layout(bargap=0.1,
                        bargroupgap = 0,
                        template='ggplot2',
                        yaxis_title="Liquidity [$]"
                    )
    fig.show()
    fig.write_html("figures/liquidity.html")

def daily_stock_price(stock, quantile=0.98):
    year_start = stock.Date.dt.year.min()
    year_end = stock.Date.dt.year.max()
    pio.renderers.default = "notebook_connected"
    fig = px.bar(stock, x='Date', y='Open', color='Yearly Percentile', title=f"Daily Stock Price for the $AAPL stock between {year_start} and {year_end}", \
                    color_discrete_map={
                    f"Top {int(100-quantile*100)}%": 'rgb(180,37,30)',
                    f"Lower {int(quantile*100)}%": 'rgb(30,50,155)'
                }
    )
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
    fig.update_traces(marker_line_width = 0,
                    selector=dict(type="bar"))

    fig.update_layout(bargap=0.1,
                    bargroupgap = 0,
                    template='ggplot2',
                    yaxis_title='Stock price [$]'
                    )
    fig.show()
    fig.write_html("figures/stock_price.html")

def daily_quotes(quotes, quantile = 0.98):
    daily_quotes = pd.DataFrame(quotes.groupby(quotes.date.dt.date).quotation.count())
    daily_quotes.index.rename('Date')
    daily_quotes.reset_index(inplace=True)
    daily_quotes.rename({'date': 'Date'}, axis=1, inplace=True)
    daily_quotes['Date']= pd.to_datetime(daily_quotes['Date'], errors='coerce')
    daily_quotes['Yearly Percentile'] = daily_quotes.apply(lambda x: x['quotation'] > np.quantile(daily_quotes[daily_quotes.Date.dt.year == x.Date.year]['quotation'], q = quantile), axis=1)
    daily_quotes['Yearly Percentile'] = daily_quotes['Yearly Percentile'].apply(lambda x : f"Top {int(100-quantile*100)}%" if x else f"Lower {int(quantile*100)}%")
    pio.renderers.default = "notebook_connected"

    year_start = quotes.date.dt.year.min()
    year_end = quotes.date.dt.year.max()
    fig = px.bar(daily_quotes, x='Date', y='quotation', color='Yearly Percentile', title=f"Daily Number of quotes related to Apple between {year_start} and {year_end}",
                    color_discrete_map={
                    f"Top {int(100-quantile*100)}%": 'rgb(180,37,30)',
                    f"Lower {int(quantile*100)}%": 'rgb(30,50,155)'
                }
    )
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
    fig.update_traces(marker_line_width = 0,
                    selector=dict(type="bar"))

    fig.update_layout(bargap=0.1,
                    bargroupgap = 0,
                    template = 'ggplot2',
                    yaxis_title="Quotations count"
                    )
    fig.show()
    fig.write_html("figures/daily_quotes.html")


    quantile = 0.98

def seasonal_analysis(df, column="Liquidity"):
    best_p_value = 1
    analysis = df.copy()
    analysis = analysis[column]
    for period in tqdm(range(5,125)):
        decompose_result_mult = seasonal_decompose(analysis, model="additive",period=period)
        residual = decompose_result_mult.resid

        p_value = adfuller(residual.dropna())[1]

        if p_value < best_p_value:
            best_period = period
            best_p_value = p_value

    print(f"The {column} can be fitted with a seasonal model of period {best_period} with p_value {best_p_value}")


    decompose_result_mult = seasonal_decompose(analysis, model="additive",period=best_period)

    trend = decompose_result_mult.trend
    seasonal = decompose_result_mult.seasonal
    residual = decompose_result_mult.resid

    seasonal = pd.DataFrame({'date': df.Date, 'trend': trend, 'seasonal': seasonal, 'residual': residual})
    year_start = seasonal.date.dt.year.min()
    year_end = seasonal.date.dt.year.max()
    pio.renderers.default = "notebook_connected"

    fig = make_subplots(rows=3, cols=1, subplot_titles=("Trend", "Seasonal Component", "Model residuals"), )

    fig.append_trace(go.Scatter(
        x=seasonal.date,
        y=seasonal.trend,
        marker = dict(color = 'rgb(30, 50, 155)')
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=seasonal.date,
        y=seasonal.seasonal,
        marker = dict(color = 'rgb(25,125,35)')
    ), row=2, col=1)

    fig.append_trace(go.Scatter(
        x=seasonal.date,
        y=seasonal.residual,
        marker = dict(color = 'rgb(180,37,30)')
    ), row=3, col=1)

    fig.update_layout(title_text=f"Fitting of {column} with seasonal component of period {best_period} days",
                    showlegend=False,
                    template = 'ggplot2'
    )
    fig['layout']['yaxis2']['title']=f'{column} [$]'
    fig['layout']['yaxis1']['title']=f'{column} [$]'
    fig['layout']['yaxis3']['title']=f'{column} [$]'

    fig.show()
    fig.write_html("figures/seasonal_analysis.html")

def stock_price_with_quotes(stock, quotes, quantile = 0.98):
    daily_quotes = pd.DataFrame(quotes.groupby(quotes.date.dt.date).quotation.count())
    daily_quotes.index.rename('Date')
    daily_quotes.reset_index(inplace=True)
    daily_quotes.rename({'date': 'Date'}, axis=1, inplace=True)
    daily_quotes['Date']= pd.to_datetime(daily_quotes['Date'], errors='coerce')
    daily_quotes['Yearly Percentile'] = daily_quotes.apply(lambda x: x['quotation'] > np.quantile(daily_quotes[daily_quotes.Date.dt.year == x.Date.year]['quotation'], q = quantile), axis=1)
    daily_quotes['Yearly Percentile'] = daily_quotes['Yearly Percentile'].apply(lambda x : f"Top {int(100-quantile*100)}%" if x else f"Lower {int(quantile*100)}%")

    year_start = quotes.date.dt.year.min()
    year_end = quotes.date.dt.year.max()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=daily_quotes['Date'], 
                            y=daily_quotes.quotation, 
                            name = "Number of quotations",
                            marker = dict(color = 'rgb(30, 50, 155)')
                            )
    )
    fig.add_trace(go.Scatter(x=stock['Date'], 
                            y=stock['Open'], 
                            name = f"AAPL stock price",
                            marker = dict(color = 'rgb(180,37,30)')
                            ),
                secondary_y=True
    )
    fig.update_traces(marker_line_width = 0,
                    selector=dict(type="bar"))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Quotations count", secondary_y=False)
    fig.update_yaxes(title_text="Stock price [$]", secondary_y=True)
    fig.update_layout(bargap=0.1,
                    bargroupgap = 0,
                    title=f"Stock price of $AAPL compared to the number of quotations related to Apple from {year_start} to {year_end}.",
                    template = 'ggplot2'
                    )
    #fig = go.Figure(data=data, layout=layout)
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
    # IPython notebook
    fig.show()
    fig.write_html("figures/daily_quotes_related_Apple_stock.html")

def pearson_stock_quotes(stock,quotes):
    stock = stock.rename({'Date':'date'},axis=1)
    quotes = pd.DataFrame(quotes.groupby(quotes.date.dt.date).quotation.count())
    quotes.reset_index(inplace=True)
    quotes['date']= pd.to_datetime(quotes['date'], errors='coerce')
    intersection = pd.Index(set(stock.date.dt.date).intersection(set(quotes.date.dt.date)))
    stock = stock[stock.date.dt.date.isin(intersection)]
    quotes = quotes[quotes.date.dt.date.isin(intersection)]

    correlation, p_value = pearsonr(stock['Liquidity'],quotes['quotation'])

    return correlation, p_value
