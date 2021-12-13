from prophet import Prophet
import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.io as pio
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
# Python
import itertools
import numpy as np
import pandas as pd


def task4(stock):
  quotes_sentiment = pd.read_pickle("data/quotes_score.pkl")
  quotes_sentiment = quotes_sentiment.groupby(quotes_sentiment.date.dt.date).sum()
  quotes_sentiment.reset_index(inplace=True)
  quotes_sentiment['date']= pd.to_datetime(quotes_sentiment['date'], errors='coerce')

  quotes_sentiment

  stock_ = stock.copy()
  stock_['diff'] = stock_.Close-stock_.Open
  stock_ = stock_[stock_.Date.dt.year.isin(range(2014,2018))]
  stock_.rename({'Date':'date','Open':'open'},inplace=True,axis=1)

  intersection = pd.Index(set(stock_.date.dt.date).intersection(set(quotes_sentiment.date.dt.date)))
  stock_to_keep = stock_[stock_.date.dt.date.isin(intersection)]
  quotes_sentiment_to_keep = quotes_sentiment[quotes_sentiment.date.dt.date.isin(intersection)]

  prediction_set = stock_to_keep.merge(quotes_sentiment_to_keep,on="date")

  ####

  m = Prophet()
  prediction_set.rename({'date':'ds','Liquidity':'y'},axis=1,inplace=True)
  m.add_regressor('positive')
  m.add_regressor('negative')
  m.add_regressor('total')
  m.fit(prediction_set)
  pio.renderers.default = "notebook_connected"

  future = m.make_future_dataframe(periods=365)
  future = future.merge(quotes_sentiment,left_on='ds',right_on='date',)
  future = future[['ds','positive','negative','total']]


  forecast = m.predict(future)
  forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


  ####
  plot_plotly(m, forecast)
  ####

  df_cv = cross_validation(m, initial='150 days', period='30 days', horizon = '60 days',parallel="processes",disable_tqdm=True)
  ###
  df_p = performance_metrics(df_cv)
  fig = plot_cross_validation_metric(df_cv, metric='mape')
  ###
  param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
  }

  # Generate all combinations of parameters
  all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
  mapes = []  # Store the RMSEs for each params here

  # Use cross validation to evaluate all parameters
  for params in tqdm(all_params):
      m = Prophet(**params).fit(df)  # Fit model with given params
      df_cv = cross_validation(m, initial='150 days', period='30 days', horizon = '60 days',parallel="processes")
      df_p = performance_metrics(df_cv, rolling_window=1)
      mapes.append(df_p['mape'].values[0])

  # Find the best parameters
  tuning_results = pd.DataFrame(all_params)
  tuning_results['mape'] = mapes
  print(tuning_results)

# changepoint_prior_scale  seasonality_prior_scale      rmse
# 0.500                     0.01  0.065398

  return None
