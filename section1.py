# Imports you may need
import seaborn as sns
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np
import yfinance as yf
import json
import bz2
from dataloader import *

def section1a(df):
    apple_data = yf.Ticker("AAPL")
    apple_data.history(period='max')