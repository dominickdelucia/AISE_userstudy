import pandas as pd
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import json

py.init_notebook_mode(connected=False)

izip = zip

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/globe_contours.csv')
df.head()

contours = []

scl = ['rgb(213,62,79)', 'rgb(244,109,67)', 'rgb(253,174,97)', \
       'rgb(254,224,139)', 'rgb(255,255,191)', 'rgb(230,245,152)', \
       'rgb(171,221,164)', 'rgb(102,194,165)', 'rgb(50,136,189)']


def pairwise(iterable):
    a = iter(iterable)
    return izip(a, a)


i = 0
for lat, lon in pairwise(df.columns):
    contours.