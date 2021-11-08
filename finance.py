import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import warnings
warnings.simplefilter(action='ignore')

def stock(stock_name):
  print("We first load the dataset")
  stock = yf.Ticker(stock_name).history(start='2019-01-01', end='2019-12-31')
  stock.reset_index(inplace=True)

  print(stock)

  print("We observe the stock price and volume during the year")

  ax1 = sns.set_style(style="white", rc=None )
  fig, ax1 = plt.subplots(figsize=(12,6))

  sns.lineplot(data = stock['Open'], sort = False, ax=ax1)
  ax2 = ax1.twinx()
  sns.barplot(y = stock['Volume'], x = stock['Date'],  alpha=0.3, ax=ax2, color="Green")
  x_dates = stock['Date'].dt.strftime('%Y-%m').sort_values().unique()
  plt.title(f"Daily stock price and volume for the ${stock_name} stock.")
  ax1.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
  ax1.set_xticklabels(labels = x_dates, rotation=45, ha='right')
  plt.show()
  print("We can also observe the distribution of the volume and daily price difference between\
    Open and Closing of the market.")

  stock['Daily Diff']=(stock['Close']-stock['Open'])*100/stock['Open']
  fig, ax3 = plt.subplots(figsize=(12,6))
  stock['Daily Diff'].plot.hist(bins=30, ax=ax3)
  plt.title(f"Histogram of the daily difference between opening and closing price for the ${stock_name} stock.")
  plt.xlabel("Percentage difference between the closing and opening price")
  plt.show()

  fig, ax4 = plt.subplots(figsize=(12,6))
  stock['Volume'].plot.hist(bins=30, ax=ax4)
  plt.title(f"Histogram of the daily volume for the ${stock_name} stock.")
  plt.xlabel("Daily volume of exchange")
  plt.show()

def compare(stock1_name, stock2_name):
  print("We observe the stock price during the year")
  stock1 = yf.Ticker(stock1_name).history(start='2019-01-01', end='2019-12-31')
  stock1.reset_index(inplace=True)
  stock2 = yf.Ticker(stock2_name).history(start='2019-01-01', end='2019-12-31')
  stock2.reset_index(inplace=True)

  fig, ax = plt.subplots(figsize=(12,6))
  sns.lineplot(data = stock1['Open'], sort = False, ax=ax, color = "red")
  ax2 = ax.twinx()
  x_dates = stock1['Date'].dt.strftime('%Y-%m').sort_values().unique()
  sns.lineplot(data = stock2['Open'], sort = False, ax=ax2)
  fig.legend([stock1_name, stock2_name], loc = "upper left", bbox_to_anchor=(0.15, 0.85, 0, 0))
  ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
  ax.set_xticklabels(labels = x_dates, rotation=45, ha='right')
  plt.show()

  print("Volume")

  fig, ax = plt.subplots(figsize=(12,6))
  sns.barplot(y = stock1['Volume'], x = stock1['Date'],  alpha=1, ax=ax, color="Red")
  ax2 = ax.twinx()
  x_dates = stock1['Date'].dt.strftime('%Y-%m').sort_values().unique()
  sns.barplot(y = stock2['Volume'], x = stock2['Date'],  alpha=0.9, ax=ax2, color="Blue")

  colors = {f'{stock1_name}':'red', f'{stock2_name}':'blue'}         
  labels = list(colors.keys())
  handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
  plt.legend(handles, labels, loc = "upper left", bbox_to_anchor=(0.03, 0.93, 0, 0))
  ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
  ax.set_xticklabels(labels = x_dates, rotation=45, ha='right')
  plt.show()

  print("histogram")
  stock1['Daily Diff']=(stock1['Close']-stock1['Open'])*100/stock1['Open']
  stock2['Daily Diff']=(stock2['Close']-stock2['Open'])*100/stock2['Open']
  fig, ax = plt.subplots(figsize=(12,6))
  ymax = max( np.max(np.histogram(stock1['Daily Diff'], bins = 30)[0]), np.max(np.histogram(stock2['Daily Diff'], bins = 30)[0]))
  ymax += ymax/12
  stock1['Daily Diff'].plot.hist(bins=30, ax=ax, color = "Red", alpha = 0.5, ylim = (0, ymax))
  ax2 = ax.twinx()
  stock2['Daily Diff'].plot.hist(bins=30, ax=ax2, color = "Blue", alpha = 0.5, ylim = (0, ymax))
  plt.legend(handles, labels, loc = "upper left", bbox_to_anchor=(0.03, 0.93, 0, 0))
  plt.title(f"Histogram of the daily difference between opening and closing price for the {stock1_name} and {stock2_name} stock.")
  plt.xlabel(f"Percentage difference between the closing and opening price for the {stock1_name} and {stock2_name}.")
  plt.show()

  # print("maxence")
  # fig, ax = plt.subplots(figsize=(12,6))
  # sns.lineplot(data = stock1['Daily Diff'], sort = False, ax=ax, color = "red")
  # ax2 = ax.twinx()
  # x_dates = stock1['Date'].dt.strftime('%Y-%m').sort_values().unique()
  # sns.lineplot(data = stock2['Daily Diff'], sort = False, ax=ax2)
  # fig.legend([stock1_name, stock2_name], loc = "upper left", bbox_to_anchor=(0.15, 0.85, 0, 0))
  # ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
  # ax.set_xticklabels(labels = x_dates, rotation=45, ha='right')

  # plt.show()

if __name__ == "__main__":
  applestock()
  


