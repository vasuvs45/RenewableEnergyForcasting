import plotly.graph_objs as go
import plotly.offline as opy
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def DataPlot(df):
    x = df['Date'].tolist()
    y = df['Solar'].tolist()
    trace1 = go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        name='Line 1'
    )
    data = [trace1]
    layout = go.Layout(
        title='Data Line Chart',        
    )
    fig = go.Figure(data=data, layout=layout)

    # create a Plotly HTML div string
    div = opy.plot(fig, auto_open=False, output_type='div')
    return div

def Naive(df):
    df_npm = df.copy()
    df_npm['lag1'] = df_npm['Solar'].shift(1)
    train_size = int(df_npm.shape[0]*0.8)
    train, test = df_npm[1:train_size],df_npm[train_size:]  

    mse = round(mean_squared_error(test['Solar'], test['lag1']),4)
    mae = round(mean_absolute_error(test['Solar'], test['lag1']),4)
    r2 = round(r2_score(test['Solar'], test['lag1']),4)
    rmse = round(mean_squared_error(test['Solar'], test['lag1'], squared=False),4)
    mape = round(np.mean(np.abs((test['Solar'] - test['lag1']) / test['Solar']) * 100),2)
    smape = round(np.mean(2 * np.abs(test['Solar'] - test['lag1']) / (np.abs(test['Solar']) + np.abs(test['lag1']))) * 100,2)

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape
    }

    x = df['Date'].tolist()
    x2 = test['Date'].tolist()
    y1 = df.Solar.tolist()
    y2 = test.lag1.tolist()
    trace1 = go.Scatter(
        x=x,
        y=y1,
        mode='lines+markers',
        name='Original'
    )
    trace2 = go.Scatter(
        x=x2,
        y=y2,
        mode='lines+markers',
        name='Naive Prediction'
    )
    data = [trace1, trace2]
    layout = go.Layout(
        title='Naive Model',        
    )
    fig = go.Figure(data=data, layout=layout)

    # create a Plotly HTML div string
    div = opy.plot(fig, auto_open=False, output_type='div')
    return div, metrics

def AR(df):
    df_AR = df.copy()
    df_AR.index = df_AR.Date
    df_AR = df_AR['Solar']
    train_size = int(df.shape[0]*0.8)
    train, test = df_AR[:train_size], df_AR[train_size:]
    modelAR = AutoReg(train, lags=10).fit()
    predictions = modelAR.predict(start=len(train), end=len(train)+len(test)-1)

    mse = round(mean_squared_error(test, predictions),4)
    mae = round(mean_absolute_error(test, predictions),4)
    r2 = round(r2_score(test, predictions),4)
    rmse = round(mean_squared_error(test, predictions, squared=False),4)
    mape = round(np.mean(np.abs((test - predictions / test)) * 100),2)
    smape = round(np.mean(2 * np.abs(test - predictions) / (np.abs(test) + np.abs(predictions))) * 100,2)

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape
    }

    x = df['Date'].tolist()
    x2 = test.index.tolist()
    y1 = df.Solar.tolist()
    y2 = predictions.tolist()
    trace1 = go.Scatter(
        x=x,
        y=y1,
        mode='lines+markers',
        name='Original'
    )
    trace2 = go.Scatter(
        x=x2,
        y=y2,
        mode='lines+markers',
        name='AutoRegression Prediction'
    )
    data = [trace1, trace2]
    layout = go.Layout(
        title='AutoRegression Model',        
    )
    fig = go.Figure(data=data, layout=layout)

    # create a Plotly HTML div string
    div = opy.plot(fig, auto_open=False, output_type='div')
    return div, metrics


def ARIMA_vis(df):
    df_ARIMA = df.copy()
    train_size = int(df.shape[0]*0.8)
    train, test = df_ARIMA[0:train_size], df_ARIMA[train_size:]
    model = ARIMA(train['Solar'], order=(10,1,7)).fit()
    predictions = model.forecast(len(test))

    mse = round(mean_squared_error(test['Solar'], predictions),4)
    mae = round(mean_absolute_error(test['Solar'], predictions),4)
    r2 = round(r2_score(test['Solar'], predictions),4)
    rmse = round(mean_squared_error(test['Solar'], predictions, squared=False),4)
    mape = round(np.mean(np.abs((test['Solar'] - predictions / test['Solar'])) * 100),2)
    smape = round(np.mean(2 * np.abs(test['Solar'] - predictions) / (np.abs(test['Solar']) + np.abs(predictions))) * 100,2)

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape
    }

    x = df['Date'].tolist()
    x2 = test.Date.tolist()
    y1 = df.Solar.tolist()
    y2 = predictions.tolist()
    trace1 = go.Scatter(
        x=x,
        y=y1,
        mode='lines+markers',
        name='Original'
    )
    trace2 = go.Scatter(
        x=x2,
        y=y2,
        mode='lines+markers',
        name='ARIMA Prediction'
    )
    data = [trace1, trace2]
    layout = go.Layout(
        title='ARIMA Model',        
    )
    fig = go.Figure(data=data, layout=layout)

    # create a Plotly HTML div string
    div = opy.plot(fig, auto_open=False, output_type='div')
    return div, metrics

def SARIMA(df):
    df_SARIMA = df.copy()
    train_size = int(df.shape[0]*0.8)
    train, test = df_SARIMA[0:train_size], df_SARIMA[train_size:]
    model_SR = SARIMAX(train['Solar'], order=(10,1,7), seasonal_order=(1,1,1,12)).fit()
    predictions = model_SR.forecast(len(test))

    mse = round(mean_squared_error(test['Solar'], predictions),4)
    mae = round(mean_absolute_error(test['Solar'], predictions),4)
    r2 = round(r2_score(test['Solar'], predictions),4)
    rmse = round(mean_squared_error(test['Solar'], predictions, squared=False),4)
    mape = round(np.mean(np.abs((test['Solar'] - predictions / test['Solar'])) * 100),2)
    smape = round(np.mean(2 * np.abs(test['Solar'] - predictions) / (np.abs(test['Solar']) + np.abs(predictions))) * 100,2)

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape
    }

    x = df['Date'].tolist()
    x2 = test.Date.tolist()
    y1 = df.Solar.tolist()
    y2 = predictions.tolist()
    trace1 = go.Scatter(
        x=x,
        y=y1,
        mode='lines+markers',
        name='Original'
    )
    trace2 = go.Scatter(
        x=x2,
        y=y2,
        mode='lines+markers',
        name='SARIMA Prediction'
    )
    data = [trace1, trace2]
    layout = go.Layout(
        title='SARIMA Model',        
    )
    fig = go.Figure(data=data, layout=layout)

    # create a Plotly HTML div string
    div = opy.plot(fig, auto_open=False, output_type='div')
    return div, metrics