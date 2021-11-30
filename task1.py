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
import seaborn as sns

def task1(quotes):

  stock_name = "AAPL"
  year = 2019
  # Find the days of high volatility 
  stock = yf.download(stock_name, start=f'{year}-01-01', end=f'{year}-12-31', progress = False)

  q1 = 0.98
  q2 = 0.9
  stock['Volatile'] = stock['Volume'] > np.quantile(stock['Volume'], q = q1)

  stock.reset_index(inplace=True)

  ax1 = sns.set_style(style="white", rc=None )
  fig, ax1 = plt.subplots(figsize=(12,6))

  #sns.lineplot(data = stock['Open'], sort = False, ax=ax1)
  ax2 = ax1.twinx()

  sns.barplot(y = stock['Volume'], x = stock['Date'],  alpha=0.3, ax=ax2, color="Green")
  sns.barplot(y = stock[stock.Volatile]['Volume'], x= stock['Date'], ax = ax2, color = 'Red')
  plt.title(f"Daily Volume for the ${stock_name} stock in {year}")

  ax1.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
  x_dates = stock['Date'].dt.strftime('%Y-%m').sort_values().unique()
  ax1.set_xticklabels(labels = x_dates, rotation=45, ha='right')

  plt.show()

  daily_quotes = pd.DataFrame(quotes.groupby(quotes.date.dt.date).quotation.count())
  daily_quotes['HighCount'] = daily_quotes['quotation'] > np.quantile(daily_quotes['quotation'], q = q2)
  daily_quotes.index.rename('Date')

  ax1 = sns.set_style(style="white", rc=None )
  fig, ax1 = plt.subplots(figsize=(12,6))
  sns.barplot(x= daily_quotes.index, y= daily_quotes['quotation'], ax = ax1, color='Green')
  #sns.barplot(x= daily_quotes.index, y= daily_quotes[daily_quotes.HighCount]['quotation'], ax = ax1, color='red')
  ax1.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
  x_dates = stock['Date'].dt.strftime('%Y-%m').sort_values().unique()
  ax1.set_xticklabels(labels = x_dates, rotation=45, ha='right')

  return None