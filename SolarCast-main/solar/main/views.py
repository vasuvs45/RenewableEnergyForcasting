from django.shortcuts import render
import plotly.graph_objs as go
import plotly.offline as opy

import pandas as pd
from . import GE

# Create your views here.
def index(response):
    return render(response, "main/germany.html", {})

def germany(request):
    df = pd.read_csv("main/data/months_germany.csv",header=0,parse_dates=[0])
    df = df[['Date','Solar']]
    min = df['Solar'].min()
    min_digit = len(str(min))
    min_digit -= 1
    df['Solar'] = df['Solar']/(10 ** min_digit)
    div1 = GE.DataPlot(df)
    div2, naive_metrics = GE.Naive(df)
    div3, ar_metrics = GE.AR(df)
    div4, arima_metrics = GE.ARIMA_vis(df)
    div5, sarima_metrics = GE.SARIMA(df)
    # pass the div string to the template context
    context = {'plot_div': div1, 'naive':div2, 'AR':div3, 'ARIMA':div4, 'SARIMA':div5, 'naive_metrics':naive_metrics, 'ar_metrics':ar_metrics, 'arima_metrics':arima_metrics, 'sarima_metrics':sarima_metrics}
    return render(request, 'main/germany.html', context)

def portugal(request):
    df = pd.read_csv("main/data/months_portugal.csv",header=0,parse_dates=[0])
    df = df[['Date','Solar']]
    min = df['Solar'].min()
    min_digit = len(str(min))
    min_digit -= 1
    df['Solar'] = df['Solar']/(10 ** min_digit)
    div1 = GE.DataPlot(df)
    div2, naive_metrics = GE.Naive(df)
    div3, ar_metrics = GE.AR(df)
    div4, arima_metrics = GE.ARIMA_vis(df)
    div5, sarima_metrics = GE.SARIMA(df)
    # pass the div string to the template context
    context = {'plot_div': div1, 'naive':div2, 'AR':div3, 'ARIMA':div4, 'SARIMA':div5, 'naive_metrics':naive_metrics, 'ar_metrics':ar_metrics, 'arima_metrics':arima_metrics, 'sarima_metrics':sarima_metrics}
    return render(request, 'main/portugal.html', context)

def belgium(request):
    df = pd.read_csv("main/data/months_belgium.csv",header=0,parse_dates=[0])
    df = df[['Date','Solar']]
    min = df['Solar'].min()
    min_digit = len(str(min))
    min_digit -= 1
    df['Solar'] = df['Solar']/(10 ** min_digit)
    div1 = GE.DataPlot(df)
    div2, naive_metrics = GE.Naive(df)
    div3, ar_metrics = GE.AR(df)
    div4, arima_metrics = GE.ARIMA_vis(df)
    div5, sarima_metrics = GE.SARIMA(df)
    # pass the div string to the template context
    context = {'plot_div': div1, 'naive':div2, 'AR':div3, 'ARIMA':div4, 'SARIMA':div5, 'naive_metrics':naive_metrics, 'ar_metrics':ar_metrics, 'arima_metrics':arima_metrics, 'sarima_metrics':sarima_metrics}
    return render(request, 'main/belgium.html', context)


def netherlands(request):
    df = pd.read_csv("main/data/months_netherlands.csv",header=0,parse_dates=[0])
    df = df[['Date','Solar']]
    # df = df[df['Solar'] != 0]
    # df = df.iloc[2:]
    min = df['Solar'].min()
    min_digit = len(str(min))
    min_digit -= 1
    min_digit = 5
    df['Solar'] = df['Solar']/(10 ** min_digit)
    div1 = GE.DataPlot(df)
    div2, naive_metrics = GE.Naive(df)
    div3, ar_metrics = GE.AR(df)
    div4, arima_metrics = GE.ARIMA_vis(df)
    div5, sarima_metrics = GE.SARIMA(df)
    # pass the div string to the template context
    context = {'plot_div': div1, 'naive':div2, 'AR':div3, 'ARIMA':div4, 'SARIMA':div5, 'naive_metrics':naive_metrics, 'ar_metrics':ar_metrics, 'arima_metrics':arima_metrics, 'sarima_metrics':sarima_metrics}
    return render(request, 'main/netherlands.html', context)