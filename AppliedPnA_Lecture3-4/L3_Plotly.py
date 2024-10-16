import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.io as pio
pio.renderers.default = "colab"

df = pd.read_csv(
    'https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

fig = go.Figure(data=[
        go.Candlestick(
            x=df['Date'],
            open=df['AAPL.Open'],
            high=df['AAPL.High'],
            low=df['AAPL.Low'],
            close=df['AAPL.Close'])])
fig.show()