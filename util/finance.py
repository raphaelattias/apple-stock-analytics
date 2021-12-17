import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import numpy as np
import warnings
warnings.simplefilter(action='ignore')
from util.plots import *
from util.quotebankexploration import *
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
pio.renderers.default = "notebook_connected"

def load_stock(stock_name = "AAPL",year_start = 2008,year_end= 2020):
  """ Load a certain stock into a pd.Dataframe

  Args:
      stock_name (str, optional): Stock name. Defaults to "AAPL".
      year_start (int, optional): Starting year. Defaults to 2008.
      year_end (int, optional): Ending year. Defaults to 2020.

  Returns:
      pd.Dataframe: Dataframe of stock features
  """
  stock = yf.download(stock_name, start=f'{year_start}-09-01', end=f'{year_end}-05-31', progress = False)
  stock.reset_index(inplace=True)
  stock['Liquidity'] = stock['Volume']*(stock['Close']+stock['Open'])/2

  return stock

def high_volatility(stock, quantile=0.98):
  """ Adds a feature Yearly Percentile to the stock dataframe when the liquidity is higher than the yearly 0.98 quantile

  Args:
      stock (pd.Datframe): Dataframe provided by yFinance of various stock features.
      quantile (float, optional): Quantile. Defaults to 0.98.

  Returns:
      pd.Dataframe: stock dataframe with additional feature
  """
  stock['Yearly Percentile'] = stock.apply(lambda x: x['Liquidity'] > np.quantile(stock[stock.Date.dt.year == x.Date.year]['Liquidity'], q = quantile), axis=1)
  stock['Yearly Percentile'] = stock['Yearly Percentile'].apply(lambda x : f"Top {int(100-quantile*100)}%" if x else f"Lower {int(quantile*100)}%")

  return stock

def weekly_liquidity(stock, quantile=0.98):
  """Plot the liquidity of a stock with additional markers for the days of high liquidity

  Args:
      stock (pd.Datframe): Dataframe provided by yFinance of various stock features.
      quantile (float, optional): Quantile. Defaults to 0.98.
  """
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
  """Plot the daily stock price with additional markers for the days of high liquidity.

  Args:
      stock (pd.Datframe): Dataframe provided by yFinance of various stock features.
      quantile (float, optional): Quantile. Defaults to 0.98.
  """
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
  """ Plot a figure of daily number of quotes along with markers on days with quotation count greater than the quantile.

  Args:
      quotes (pd.Datframe): Dataframe of quotes.
      quantile (float, optional): Quantile. Defaults to 0.98.
  """
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
  """Performs seasonal analysis on a dataframe. This method is useful to find the best fitting seasonal pattern of time series.

  Args:
      df (pd.Dataframe): Time series dataframe
      column (str, optional): Feature to performs the analysis. Defaults to "Liquidity".
  """
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
  """ Plot figure of daily stock price along with the quotes and markers for days of high volatility

  Args:
      stock (pd.Datframe): Dataframe provided by yFinance of various stock features.
      quotes (pd.Datframe): Dataframe of quotes.
  """
  daily_quotes = pd.DataFrame(quotes.groupby(quotes.date.dt.date).quotation.count())
  daily_quotes.index.rename('Date')
  daily_quotes.reset_index(inplace=True)
  daily_quotes.rename({'date': 'Date'}, axis=1, inplace=True)
  daily_quotes['Date']= pd.to_datetime(daily_quotes['Date'], errors='coerce')
  daily_quotes['Yearly Percentile'] = daily_quotes.apply(lambda x: x['quotation'] > np.quantile(daily_quotes[daily_quotes.Date.dt.year == x.Date.year]['quotation'], q = quantile), axis=1)
  daily_quotes['Yearly Percentile'] = daily_quotes['Yearly Percentile'].apply(lambda x : f"Top {int(100-quantile*100)}%" if x else f"Lower {int(quantile*100)}%")

  daily_quotes['quotation'] = daily_quotes['quotation'].rolling(7, min_periods=1).mean().to_frame()

  intersection = pd.Index(set(stock.Date.dt.date).intersection(set(daily_quotes.Date.dt.date)))
  stock = stock[stock.Date.isin(intersection)]
  daily_quotes = daily_quotes[daily_quotes.Date.isin(intersection)]

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
  fig.update_yaxes(title_text="Quotations count", secondary_y=False, range=[0,1100])
  fig.update_yaxes(title_text="Stock price [$]", secondary_y=True, range=[0,90])
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
  """Compute the correlation between the stock liquidity and the daily number of quotes.

  Args:
      stock (pd.Datframe): Dataframe provided by yFinance of various stock features.
      quotes (pd.Datframe): Dataframe of quotes.

  Returns:
      (correlation, p-value): Correlations between the two time series along with the p-value.
  """
  stock = stock.rename({'Date':'date'},axis=1)
  quotes = pd.DataFrame(quotes.groupby(quotes.date.dt.date).quotation.count())
  quotes.reset_index(inplace=True)
  quotes['date']= pd.to_datetime(quotes['date'], errors='coerce')
  intersection = pd.Index(set(stock.date.dt.date).intersection(set(quotes.date.dt.date)))
  stock = stock[stock.date.dt.date.isin(intersection)]
  quotes = quotes[quotes.date.dt.date.isin(intersection)]

  correlation, p_value = pearsonr(stock['Liquidity'],quotes['quotation'])

  return correlation, p_value


def stock(stock_name = 'AAPL', year = 2020, fig = "price_volume"):
  """ Set of figures to plot to performs some first analysis of a stock

  Args:
      stock_name ('string'): Stock ticker. Defaults to 'AAPL'
      year (int, optional): Year to perform analysis on. Defaults to 2020.
      fig (str, optional): Figure to plot. Defaults to "price_volume".
  """
  assert fig in ["price_volume", "daily_diff", "volume"]

  stock = yf.download(stock_name, start=f'{year}-01-01', end=f'{year}-12-31', progress = False)
  stock.reset_index(inplace=True)

  if fig == "price_volume":
    ax1 = sns.set_style(style="white", rc=None )
    fig, ax1 = plt.subplots(figsize=(12,6))

    sns.lineplot(data = stock['Open'], sort = False, ax=ax1)
    ax2 = ax1.twinx()

    sns.barplot(y = stock['Volume'], x = stock['Date'],  alpha=0.3, ax=ax2, color="Green")

    plt.title(f"Daily stock price and volume for the ${stock_name} stock in {year}")

    ax1.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
    x_dates = stock['Date'].dt.strftime('%Y-%m').sort_values().unique()
    ax1.set_xticklabels(labels = x_dates, rotation=45, ha='right')

    plt.show()

  elif fig == "daily_diff":
    stock['Daily Diff']=(stock['Close']-stock['Open'])*100/stock['Open']
    fig, ax3 = plt.subplots(figsize=(12,6))
    stock['Daily Diff'].plot.hist(bins=30, ax=ax3, alpha = alpha_)
    plt.axvline(stock['Daily Diff'].mean(), color = "r")
    plt.title(f"Histogram of the daily difference between opening and closing price for the ${stock_name} stock in {year}.")
    plt.xlabel("Percentage difference between the closing and opening price")
    plt.show()

  elif fig == "volume":
    fig, ax4 = plt.subplots(figsize=(12,6))
    stock['Volume'].plot.hist(bins=30, ax=ax4, alpha = alpha_)
    plt.axvline(stock['Volume'].mean(), color = "r")
    plt.title(f"Histogram of the daily volume for the ${stock_name} stock in {year}.")
    plt.xlabel("Daily volume of exchange")
    plt.show()

def compare(stock1_name, stock2_name, year = 2020, fig = "price"):
  """ Set of figures to plot to performs some first comparisons between two stocks

  Args:
      stock1_name ('string'): Stock ticker. Defaults to 'AAPL'
      stock1_name ('string'): Stock ticker. Defaults to 'MST'
      year (int, optional): Year to perform analysis on. Defaults to 2020.
      fig (str, optional): Figure to plot. Defaults to "price_volume".
  """
  assert fig in ["price", "volume", "daily_diff"]

  stock1 = yf.download(stock1_name, start=f'{year}-01-01', end=f'{year}-12-31', progress = False)
  stock1.reset_index(inplace=True)
  stock2 = yf.download(stock2_name, start=f'{year}-01-01', end=f'{year}-12-31', progress = False)
  stock2.reset_index(inplace=True)

  if fig == "price":
    fig, ax = plt.subplots(figsize=(12,6))
    sns.lineplot(data = stock1['Open'], sort = False, ax=ax, color = [1, 0.50, 0.05])
    ax2 = ax.twinx()
    x_dates = stock1['Date'].dt.strftime('%Y-%m').sort_values().unique()
    sns.lineplot(data = stock2['Open'], sort = False, ax=ax2)
    fig.legend([stock1_name, stock2_name], loc = "upper left", bbox_to_anchor=(0.15, 0.85, 0, 0))
    ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
    ax.set_xticklabels(labels = x_dates, rotation=45, ha='right')
    ax.set_xlabel(f"Months of the year {year}")
    plt.title(f"Daily price at open of the {stock1_name} stock and {stock2_name} stock during the year {year}.")
    plt.show()

  if fig == "volume":
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(y = stock1['Volume'], x = stock1['Date'],  alpha=1, ax=ax, color=[1, 0.50, 0.05])
    ax2 = ax.twinx()
    x_dates = stock1['Date'].dt.strftime('%Y-%m').sort_values().unique()
    sns.barplot(y = stock2['Volume'], x = stock2['Date'],  alpha=0.9, ax=ax2, color=[0.12, 0.46, 0.70])

    colors = {f'{stock1_name}':'Orange', f'{stock2_name}':[0.12, 0.46, 0.70]}         
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels, loc = "upper left", bbox_to_anchor=(0.03, 0.93, 0, 0))
    ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
    ax.set_xticklabels(labels = x_dates, rotation=45, ha='right')
    plt.title(f"Daily volume of exchange of the {stock1_name} stock and {stock2_name} stock  \n during the year {year}.")
    plt.show()

  if fig == "daily_diff":
    stock1['Daily Diff']=(stock1['Close']-stock1['Open'])*100/stock1['Open']
    stock2['Daily Diff']=(stock2['Close']-stock2['Open'])*100/stock2['Open']
    fig, ax = plt.subplots(figsize=(12,6))
    colors = {f'{stock1_name}':'Orange', f'{stock2_name}':[0.12, 0.46, 0.70]}         
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    ymax = max( np.max(np.histogram(stock1['Daily Diff'], bins = 30)[0]), np.max(np.histogram(stock2['Daily Diff'], bins = 30)[0]))
    ymax += ymax/12
    stock1['Daily Diff'].plot.hist(bins=30, ax=ax, color = [1, 0.50, 0.05], alpha = 0.5, ylim = (0, ymax))
    ax2 = ax.twinx()
    stock2['Daily Diff'].plot.hist(bins=30, ax=ax2, color = [0.12, 0.46, 0.70], alpha = 0.5, ylim = (0, ymax))
    plt.legend(handles, labels, loc = "upper left", bbox_to_anchor=(0.03, 0.93, 0, 0))
    plt.title(f"Histogram of the daily difference between opening and closing price for the {stock1_name} and {stock2_name} stock.")
    ax.set_xlabel(f"Percentage difference between the closing and opening price for the {stock1_name} and {stock2_name}.")
    plt.show()