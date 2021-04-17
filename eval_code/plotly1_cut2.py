# Import required libraries
import os
import pickle
import copy
import datetime as dt
import math

import requests
import pandas as pd
from flask import Flask
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

# Multi-dropdown options
from controls import COUNTIES, WELL_STATUSES, WELL_TYPES, WELL_COLORS

app = dash.Dash(__name__)
server = app.server

# Create controls
county_options = [{'label': str(COUNTIES[county]), 'value': str(county)}
                  for county in COUNTIES]

well_status_options = [{'label': str(WELL_STATUSES[well_status]),
                        'value': str(well_status)}
                       for well_status in WELL_STATUSES]

well_type_options = [{'label': str(WELL_TYPES[well_type]),
                      'value': str(well_type)}
                     for well_type in WELL_TYPES]


# Load data
df = pd.read_csv('data/wellspublic.csv')
df['Date_Well_Completed'] = pd.to_datetime(df['Date_Well_Completed'])
df = df[df['Date_Well_Completed'] > dt.datetime(1960, 1, 1)]

trim = df[['API_WellNo', 'Well_Type', 'Well_Name']]
trim.index = trim['API_WellNo']
dataset = trim.