# Useful starting lines
import numpy as np
from dataloader import *
from plots import *
from finance import stock, compare
from quotebankexploration import *
from wikipedia import *
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import seaborn as sns

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import numpy as np

pio.renderers.default = "notebook_connected"

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

def task1():

  stock_name = "AAPL"
  year = 2019
  year_start = 2015
  year_end = 2019
  # Find the days of high volatility 
  stock = yf.download(stock_name, start=f'{year_start}-01-01', end=f'{year_end}-08-31', progress = False)
  stock.reset_index(inplace=True)

  quotes = pd.concat([load_quotes(i, 'processed quotes') for i in range(year_start,year_end+1)])

  q1 = 0.98
  q2 = 0.98


  stock['Normalized'] = stock['Open']*stock['Volume']
  stock['Volatile'] = stock.apply(lambda x: x['Normalized'] > np.quantile(stock[stock.Date.dt.year == x.Date.year]['Normalized'], q = q1), axis=1)



  ######


  weekly = pd.DataFrame(stock.resample('W', on='Date')['Normalized'].sum())
  weekly['Volume'] = stock.resample('W', on='Date')['Volume'].sum()
  weekly.index.rename('Date')
  weekly.reset_index(inplace=True)
  weekly['Volatile'] = weekly.apply(lambda x: x['Normalized'] > np.quantile(weekly[weekly.Date.dt.year == x.Date.year]['Normalized'], q = q1), axis=1)

  pio.renderers.default = "notebook_connected"
  fig = px.bar(stock, x='Date', y='Normalized', color='Volatile', title=f"Liquidity traded for the ${stock_name} stock between {year_start} and {year_end}")
  fig.update_xaxes(
      rangeslider_visible=True,
      rangeselector=dict(
          buttons=list([
              dict(count=1, label="1m", step="month", stepmode="backward"),
              dict(count=6, label="6m", step="month", stepmode="backward"),
              dict(count=1, label="YTD", step="year", stepmode="todate"),
              dict(count=1, label="1y", step="year", stepmode="backward"),
              dict(step="all")
          ])
      )
  )
  fig.update_traces(marker_line_width = 0,
                    selector=dict(type="bar"))

  fig.update_layout(bargap=0.1,
                    bargroupgap = 0,
                  )
  fig.show()

  ######


 
  pio.renderers.default = "notebook_connected"
  fig = px.bar(stock, x='Date', y='Open', color='Volatile', title=f"Daily Stock Price for the ${stock_name} stock between {year_start} and {year_end}")
  fig.update_xaxes(
      rangeslider_visible=True,
      rangeselector=dict(
          buttons=list([
              dict(count=1, label="1m", step="month", stepmode="backward"),
              dict(count=6, label="6m", step="month", stepmode="backward"),
              dict(count=1, label="YTD", step="year", stepmode="todate"),
              dict(count=1, label="1y", step="year", stepmode="backward"),
              dict(step="all")
          ])
      )
  )
  fig.update_traces(marker_line_width = 0,
                    selector=dict(type="bar"))

  fig.update_layout(bargap=0.1,
                    bargroupgap = 0,
                  )
  fig.show()



  ######


  daily_quotes = pd.DataFrame(quotes.groupby(quotes.date.dt.date).quotation.count())
  daily_quotes.index.rename('Date')
  daily_quotes.reset_index(inplace=True)
  daily_quotes['date']= pd.to_datetime(daily_quotes['date'], errors='coerce')
  daily_quotes['HighCount'] = daily_quotes.apply(lambda x: x['quotation'] > np.quantile(daily_quotes[daily_quotes.date.dt.year == x.date.year]['quotation'], q = q1), axis=1)
  #daily_quotes['HighCount'] = daily_quotes['quotation'] > np.quantile(daily_quotes['quotation'], q = q2)


  pio.renderers.default = "notebook_connected"
  fig = px.bar(daily_quotes, x='date', y='quotation', color='HighCount', title=f"Daily Number of quotes related to Apple between {year_start} and {year_end}")
  fig.update_xaxes(
      rangeslider_visible=True,
      rangeselector=dict(
          buttons=list([
              dict(count=1, label="1m", step="month", stepmode="backward"),
              dict(count=6, label="6m", step="month", stepmode="backward"),
              dict(count=1, label="YTD", step="year", stepmode="todate"),
              dict(count=1, label="1y", step="year", stepmode="backward"),
              dict(step="all")
          ])
      )
  )
  fig.update_traces(marker_line_width = 0,
                    selector=dict(type="bar"))

  fig.update_layout(bargap=0.1,
                    bargroupgap = 0,
                  )
  fig.show()

  # ax1 = sns.set_style(style="white", rc=None )
  # fig, ax1 = plt.subplots(figsize=(12,6))
  # daily_quotes.date = pd.to_datetime(daily_quotes.date, format='%Y-%m-%d')
  # sns.barplot(x= daily_quotes['date'], y= daily_quotes['quotation'], ax = ax1, color='green')
  # sns.barplot(x= daily_quotes['date'], y= daily_quotes[daily_quotes.HighCount]['quotation'], ax = ax1, color='red')
  # ax1.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
  # x_dates = daily_quotes['date'].dt.strftime('%Y-%m').sort_values().unique()
  # ax1.set_xticklabels(labels = x_dates, rotation=45, ha='right')

  #######
  from statsmodels.tsa.seasonal import seasonal_decompose
  from statsmodels.tsa.stattools import adfuller

  analysis = stock.copy()
  analysis.set_index('Date',inplace=True)
  analysis = analysis['Open']
  decompose_result_mult = seasonal_decompose(analysis, model="additive",period=12)

  trend = decompose_result_mult.trend
  seasonal = decompose_result_mult.seasonal
  residual = decompose_result_mult.resid

  p_value = adfuller(residual.dropna())[1]
  print(f"p-value : {p_value}")

  decompose_result_mult.plot();
######
  best_value = 1
  for period in range(1,125):
    analysis = stock.copy()
    analysis.set_index('Date',inplace=True)
    analysis = analysis['Volume']
    decompose_result_mult = seasonal_decompose(analysis, model="additive",period=period)

    trend = decompose_result_mult.trend
    seasonal = decompose_result_mult.seasonal
    residual = decompose_result_mult.resid

    p_value = adfuller(residual.dropna())[1]

    if p_value < best_value:
      best_period = period
      best_value = p_value

  print(best_period,best_value)

  return None