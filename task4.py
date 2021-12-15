from prophet import Prophet
import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.io as pio
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from sklearn.model_selection import TimeSeriesSplit
from prophet.plot import plot_cross_validation_metric
# Python
import itertools
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import plotly.graph_objects as go

import os
import sys
import logging
import warnings

def build_prediction_frame(stock,quotes_sentiment = None):

  if (quotes_sentiment).any()[0]:
    quotes_sentiment = quotes_sentiment.groupby(quotes_sentiment.date.dt.date).sum()
    quotes_sentiment.reset_index(inplace=True)
    quotes_sentiment['date']= pd.to_datetime(quotes_sentiment['date'], errors='coerce')

  stock_ = stock.copy()
  stock_.rename({'Date':'date'},inplace=True,axis=1)

  intersection = pd.Index(set(stock_.date.dt.date).intersection(set(quotes_sentiment.date.dt.date)))
  stock_to_keep = stock_[stock_.date.dt.date.isin(intersection)]
  quotes_sentiment_to_keep = quotes_sentiment[quotes_sentiment.date.dt.date.isin(intersection)]

  prediction_frame = stock_to_keep.merge(quotes_sentiment_to_keep,on="date")

  return prediction_frame

def times_series_predict(stock, quotes_sentiment = None, features = None, response = 'Open'):
  tscv = TimeSeriesSplit()
  figs = []
  prediction_frame = build_prediction_frame(stock,quotes_sentiment)

  for train_index, test_index in tscv.split(prediction_frame):
    m = Prophet(changepoint_prior_scale=1.0,seasonality_prior_scale=0.021544)
    prediction_frame_shorter = build_prediction_frame(stock.iloc[train_index],quotes_sentiment)
    m = fit_prophet(m, prediction_frame_shorter, features=features)
    pred = predict_future(m,prediction_frame,feature_frame=quotes_sentiment)
    fig = plot_plotly(m,pred)
    figs.append(fig)

def prophet_cross_validation(param_grid, stock, quotes_sentiment = None, features = None,response = 'Open', metric = 'mape'):

    # Generate all combinations of parameters
  all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
  prediction_frame = build_prediction_frame(stock,quotes_sentiment)
  mapes = []  # Store the RMSEs for each params here
  figs = []

  # Use cross validation to evaluate all parameters
  for params in tqdm(all_params):
      m = Prophet(**params)
      m = fit_prophet(m, prediction_frame, response='Open')
      df_cv = cross_validation(m, initial='150 days', period='30 days', horizon = '60 days',parallel="processes",disable_tqdm=True)
      df_p = performance_metrics(df_cv)
      mapes.append(df_p[metric].values[0])

  # Find the best parameters
  tuning_results = pd.DataFrame(all_params)
  tuning_results[metric] = mapes

  return tuning_results


def fit_prophet(m, prediction_frame, features=None, response='Open', params = None):
  prediction_frame.rename({'date':'ds', response:'y'},axis=1,inplace=True)
  if features:
    for feature in features:
      m.add_regressor(feature)    
  m.fit(prediction_frame)

  return m

def predict_future(m, prediction_frame, feature_frame = None):
    future = m.make_future_dataframe(periods=300)
    
    if feature_frame.any()[0]:
      feature_frame = feature_frame.rename({'date':'ds', 'Date':'ds'},axis=1)
      future = future.merge(feature_frame,left_on='ds',right_on='ds')

    forecast = m.predict(future)

    return forecast

def plot_prediction(stock, quotes_sentiment, pred):
    prediction_frame = build_prediction_frame(stock[stock.Date.dt.year.isin(range(2015,2019))],quotes_sentiment)
    df = prediction_frame[['date','Open']].merge(pred[['ds','yhat_lower','yhat_upper', 'yhat']],left_on="date",right_on="ds")

    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=df.ds, y=df.Open,
                        mode='markers',
                        name='True Price'))
    fig.add_trace(go.Scatter(x=df.ds, y=df.yhat,
                        mode='lines',
                        name='Predicted Price',
                        marker_color='rgb(50,90,200)'))

    fig.add_trace(go.Scatter(x=df.ds, y=df.yhat_upper,
                        mode='lines',
                        name='Lower Prediction',
                        marker_color='rgb(165,37,30)'))
                        
    fig.add_trace(go.Scatter(x=df.ds, y=df.yhat_lower,
                        mode='lines',
                        name='Upper Prediction',
                        marker_color='rgb(50,120,70)',
                        fillcolor='rgba(68, 68, 68, 0.3)',
                        fill='tonexty'))

    fig.add_vline(x='2018-01-01', line_width=2, line_dash="dash", line_color="black")

    fig.update_layout(
                    template = 'ggplot2',
                    yaxis_title="Stock Price [$]",
                    xaxis_title = "Date",
                    title = "Fitted model between 2015 and 2018, and predicted on 2019 for Apple stock price"
                    )
    fig.show()

    fig.write_html('figures/future_stock_prediction.html')

