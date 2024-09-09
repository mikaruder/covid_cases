import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from prophet import Prophet
import holidays
from xgboost import XGBRegressor as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def calculate_mape(actual, predict):
    absolute_percentage_error = np.abs((actual - predict) / actual)
    mape = round(np.mean(absolute_percentage_error) * 100,2)
    return mape