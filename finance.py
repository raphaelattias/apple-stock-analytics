import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import warnings
warnings.simplefilter(action='ignore')

def applestock():
  print("We first load the dataset")
  apple = yf.Ticker("AAPL").history(start='2019-01-01', end='2019-12-31')
  apple.reset_index(inplace=True)

  print(apple)

  print("We observe the stock price and volume during the year")

  ax1 = sns.set_style(style="white", rc=None )
  fig, ax1 = plt.subplots(figsize=(12,6))

  sns.lineplot(data = apple['Open'], sort = False, ax=ax1)
  ax2 = ax1.twinx()
  sns.barplot(y = apple['Volume'], x = apple['Date'],  alpha=0.3, ax=ax2, color="Green")
  x_dates = apple['Date'].dt.strftime('%Y-%m').sort_values().unique()
  plt.title("Daily stock price and volume for the $AAPL stock.")
  ax1.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
  ax1.set_xticklabels(labels = x_dates, rotation=45, ha='right')

  plt.show()
  print("We can also observe the distribution of the volume and daily price difference between\
    Open and Closing of the market.")

  apple['Daily Diff']=(apple['Close']-apple['Open'])*100/apple['Open']
  fig, ax3 = plt.subplots(figsize=(12,6))
  apple['Daily Diff'].plot.hist(bins=25, ax=ax3)
  plt.title("Histogram of the daily difference between opening and closing price for the $AAPL stock.")
  plt.xlabel("Percentage difference between the closing and opening price")
  plt.show()

  fig, ax4 = plt.subplots(figsize=(12,6))
  apple['Volume'].plot.hist(bins=25, ax=ax4)
  plt.title("Histogram of the daily volume for the $AAPL stock.")
  plt.xlabel("Daily volume of exchange")
  plt.show()

if __name__ == "__main__":
  applestock()
  


