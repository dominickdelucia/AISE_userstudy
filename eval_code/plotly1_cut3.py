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
dataset = trim.to_dict(orient='index')

points = pickle.load(open("data/points.pkl", "rb"))


# Create global chart template
mapbox_access_token = 'pk.eyJ1IjoiamFja2x1byIsImEiOiJjajNlcnh3MzEwMHZtMzNueGw3NWw5ZXF5In0.fk8k06T96Ml9CLGgKmk81w'

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(
        l=30,
        r=30,
        b=20,
        t=40
    ),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation='h'),
    title='Satellite Overview',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(
            lon=-78.05,
            lat=42.54
        ),
        zoom=7,
    )
)

# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id='aggregate_data'),
        html.Div(
            [
                html.Div(
                    [
                        html.H2(
                            'New York Oil and Gas',

                        ),
                        html.H4(
                            'Production Overview',
                        )
                    ],

                    className='eight columns'
                ),
                html.Img(
                    src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe.png",
                    className='two columns',
                ),
                html.A(
                    html.Button(
                        "Learn More",
                        id="learnMore"

                    ),
                    href="https://plot.ly/dash/pricing/",
                    className="two columns"
                )
            ],
            id="header",
            className='row',
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            'Filter by construction date (or select range in histogram):',
                            className="control_label"
                        ),
                        dcc.RangeSlider(
                            id='year_slider',
                            min=1960,
                            max=2017,
                            value=[1990, 2010],
                            className="dcc_control"
                        ),
                        html.P(
                            'Filter by well status:',
                            className="control_label"
                        ),
                        dcc.RadioItems(
                            id='well_status_selector',
                            options=[
                                {'label': 'All ', 'value': 'all'},
                                {'label': 'Active only ', 'value': 'active'},
                                {'label': 'Customize ', 'value': 'custom'}
                            ],
                            value='active',
                            labelStyle={'display': 'inline-block'},
                            className="dcc_control"
                        ),
                        dcc.Dropdown(
                            id='well_statuses',
                            options=well_status_options,
                            multi=True,
                            value=list(WELL_STATUSES.keys()),
                            className="dcc_control"
                        ),
                        dcc.Checklist(
                            id='lock_selector',
                            options=[
                                {'label': 'Lock camera', 'value': 'locked'}
                            ],
                            values=[],
                            className="dcc_control"
                        ),
                        html.P(
                            'Filter by well type:',
                            className="control_label"
                        ),
                        dcc.RadioItems(
                            id='well_type_selector',
                            options=[
                                {'label': 'All ', 'value': 'all'},
                                {'label': 'Productive only ',
                                    'value': 'productive'},
                                {'label': 'Customize ', 'value': 'custom'}
                            ],
                            value='productive',
                            labelStyle={'display': 'inline-block'},
                            className="dcc_control"
                        ),
                        dcc.Dropdown(
                            id='well_types',
                            options=well_type_options,
                            multi=True,
                            value=list(WELL_TYPES.keys()),
                            className="dcc_control"
                        ),
                    ],
                    className="pretty_container four columns"
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P("No. of Wells"),
                                        html.H6(
                                            id="well_text",
                                            className="info_text"
                                        )
                                    ],
                                    id="wells",
                                    className="pretty_container"
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P("Gas"),
                                                html.H6(
                                                    id="gasText",
                                                    className="info_text"
                                                )
                                            ],
                                            id="gas",
                                            className="pretty_container"
                                        ),
                                        html.Div(
                                            [
                                                html.P("Oil"),
                                                html.H6(
                                                    id="oilText",
                                                    className="info_text"
                                                )
                                            ],
                                            id="oil",
                                            className="pretty_container"
                                        ),
                                        html.Div(
                                            [
                                                html.P("Water"),
                                                html.H6(
                                                    id="waterText",
                                                    className="info_text"
                                                )
                                            ],
                                            id="water",
                                            className="pretty_container"
                                        ),
                                    ],
                                    id="tripleContainer",
                                )

                            ],
                            id="infoContainer",
                            className="row"
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id='count_graph',
                                )
                            ],
                            id="countGraphContainer",
                            className="pretty_container"
                        )
                    ],
                    id="rightCol",
                    className="eight columns"
                )
            ],
            className="row"
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id='main_graph')
                    ],
                    className='pretty_container eight columns',
                ),
                html.Div(
                    [
                        dcc.Graph(id='individual_graph')
                    ],
                    className='pretty_container four columns',
                ),
            ],
            className='row'
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id='pie_graph')
                    ],
                    className='pretty_container seven columns',
                ),
                html.Div(
                    [
                        dcc.Graph(id='aggregate_graph')
                    ],
                    className='pretty_container five columns',
                ),
            ],
            className='row'
        ),
    ],
    id="mainContainer",
    style={
        "display": "flex",
        "flex-direction": "column"
    }
)


# Helper functions
def human_format(num):

    magnitude = int(math.log(num, 1000))
    mantissa = str(int(num / (1000**magnitude)))
    return mantissa + ['', 'K', 'M', 'G', 'T', 'P'][magnitude]


def filter_dataframe(df, well_statuses, well_types, year_slider):
    dff = df[df['Well_Status'].isin(well_statuses)
             & df['Well_Type'].isin(well_types)
             & (df['Date_Well_Completed'] > dt.datetime(year_slider[0], 1, 1))
             & (df['Date_Well_Completed'] < dt.datetime(year_slider[1], 1, 1))]
    return dff


def fetch_individual(api):
    try:
        points[api]
    except:
        return None, None, None, None

    index = list(range(min(points[api].keys()), max(points[api].keys()) + 1))
    gas = []
    oil = []
    water = []

    for year in index:
        try:
            gas.append(points[api][year]['Gas Produced, MCF'])
        except:
            gas.append(0)
        try:
            oil.append(points[api][year]['Oil Produced, bbl'])
        except:
            oil.append(0)
        try:
            water.append(points[api][year]['Water Produced, bbl'])
        except:
            water.append(0)

    return index, gas, oil, water


def fetch_aggregate(selected, year_slider):

    index = list(range(max(year_slider[0], 1985), 2016))
    gas = []
    oil = []
    water = []

    for year in index:
        count_gas = 0
        count_oil = 0
        count_water = 0
        for api in selected:
            try:
                count_gas += points[api][year]['Gas Produced, MCF']
            except:
                pass
            try:
                count_oil += points[api][year]['Oil Produced, bbl']
            except:
                pass
            try:
                count_water += points[api][year]['Water Produced, bbl']
            except:
                pass
        gas.append(count_gas)
        oil.append(count_oil)
        water.append(count_water)

    return index, gas, oil, water


# Create callbacks
@app.callback(Output('aggregate_data', 'data'),
              [Input('well_statuses', 'value'),
               Input('well_types', 'value'),
               Input('year_slider', 'value')])
def update_production_text(well_statuses, well_types, year_slider):

    dff = filter_dataframe(df, well_statuses, well_types, year_slider)
    selected = dff['API_WellNo'].values
    index, gas, oil, water = fetch_aggregate(selected, year_slider)
    return [human_format(sum(gas)), human_format(sum(oil)), human_format(sum(water))]

# Radio -> multi


@app.callback(Output('well_statuses', 'value'),
              [Input('well_status_selector', 'value')])
def display_status(selector):
    if selector == 'all':
        return list(