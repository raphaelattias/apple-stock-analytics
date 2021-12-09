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
def task1():

  stock_name = "AAPL"
  year = 2019
  year_start = 2015
  year_end = 2019
  # Find the days of high volatility 
  stock = yf.download(stock_name, start=f'{year_start}-01-01', end=f'{year_end}-12-31', progress = False)
  stock.reset_index(inplace=True)

  quotes = pd.concat([load_quotes(i, 'processed quotes') for i in range(year_start,year_end+1)])
  quotes.rename({'quotation': 'Quotation'}, axis = 1, inplace=True)
  q1 = 0.98
  q2 = 0.98


  stock['Liquidity'] = stock['Open']*stock['Volume']
  stock['Volatility'] = stock.apply(lambda x: x['Liquidity'] > np.quantile(stock[stock.Date.dt.year == x.Date.year]['Liquidity'], q = q1), axis=1)
  stock['Volatility'] = stock.Volatility.apply(lambda x : "Volatile" if x else "Regular")


  ######


  weekly = pd.DataFrame(stock.resample('W', on='Date')['Liquidity'].sum())
  weekly['Volume'] = stock.resample('W', on='Date')['Volume'].sum()
  weekly.index.rename('Date')
  weekly.reset_index(inplace=True)
  weekly['Volatility'] = weekly.apply(lambda x: x['Liquidity'] > np.quantile(weekly[weekly.Date.dt.year == x.Date.year]['Liquidity'], q = q1), axis=1)

  pio.renderers.default = "notebook_connected"
  fig = px.bar(stock, x='Date', y='Liquidity', color='Volatility', title=f"Liquidity traded for the ${stock_name} stock between {year_start} and {year_end}")
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
                  )
  fig.show()
  fig.write_html("figures/liquidity.html")

  ######


 
  pio.renderers.default = "notebook_connected"
  fig = px.bar(stock, x='Date', y='Open', color='Volatility', title=f"Daily Stock Price for the ${stock_name} stock between {year_start} and {year_end}")
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
                  )
  fig.show()
  fig.write_html("figures/stock_price.html")


  ######


  daily_quotes = pd.DataFrame(quotes.groupby(quotes.date.dt.date).Quotation.count())
  daily_quotes.index.rename('Date')
  daily_quotes.reset_index(inplace=True)
  daily_quotes.rename({'date': 'Date'}, axis=1, inplace=True)
  daily_quotes['Date']= pd.to_datetime(daily_quotes['Date'], errors='coerce')
  daily_quotes['Yearly Percentile'] = daily_quotes.apply(lambda x: x['Quotation'] > np.quantile(daily_quotes[daily_quotes.Date.dt.year == x.Date.year]['Quotation'], q = q2), axis=1)
  daily_quotes['Yearly Percentile'] = daily_quotes['Yearly Percentile'].apply(lambda x : f"Top {int(100-q2*100)}%" if x else f"Lower {int(q2*100)}%")
  
  #daily_quotes['High Count'] = daily_quotes['Quotation'] > np.quantile(daily_quotes['Quotation'], q = q2)


  pio.renderers.default = "notebook_connected"
  fig = px.bar(daily_quotes, x='Date', y='Quotation', color='Yearly Percentile', title=f"Daily Number of quotes related to Apple between {year_start} and {year_end}")
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
                  )
  fig.show()
  fig.write_html("figures/daily_quotes.html")

  # ax1 = sns.set_style(style="white", rc=None )
  # fig, ax1 = plt.subplots(figsize=(12,6))
  # daily_quotes.date = pd.to_datetime(daily_quotes.date, format='%Y-%m-%d')
  # sns.barplot(x= daily_quotes['date'], y= daily_quotes['Quotation'], ax = ax1, color='green')
  # sns.barplot(x= daily_quotes['date'], y= daily_quotes[daily_quotes.High Count]['Quotation'], ax = ax1, color='red')
  # ax1.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
  # x_dates = daily_quotes['date'].dt.strftime('%Y-%m').sort_values().unique()
  # ax1.set_xticklabels(labels = x_dates, rotation=45, ha='right')

  #######

  ######

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

  ####
  # layout = go.Layout(
  #     barmode='stack',
  #     title='Stacked Bar with Pandas'
  # )


  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(go.Bar(x=daily_quotes['Date'], y=daily_quotes['Quotation'], name = "Number of quotations" ))
  fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Open'], name = f"{stock_name} stock price"),secondary_y=True)
  fig.update_traces(marker_line_width = 0,
                  selector=dict(type="bar"))
  fig.update_xaxes(title_text="Date")
  fig.update_yaxes(title_text="Quotations", secondary_y=False)
  fig.update_yaxes(title_text="Price in USD$", secondary_y=True)
  fig.update_layout(bargap=0.1,
                  bargroupgap = 0,
                  title=f"Stock price of ${stock_name} compared to the number of quotations related to Apple from {year_start} to {year_end}."
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

  ####

  stock_to_keep = stock[stock.Date.isin(set(stock.Date).intersection(set(daily_quotes.Date)))]
  daily_quotes_to_keep = daily_quotes[daily_quotes.Date.isin(set(stock.Date).intersection(set(daily_quotes.Date)))]

  pearsonr(stock_to_keep.Liquidity,daily_quotes_to_keep.Quotation)


  return None