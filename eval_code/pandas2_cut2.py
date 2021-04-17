import pandas as pd
import numpy as np
import itertools
import helpers
import preprocess


def closeprice_simplify_price_df (price_df):
    df = price_df.iloc[:,~price_df.columns.str.match('b')]
    df['idx'] = df.index
    df['timestamp'] = df.TS_END
    df['close_price'] = df.loc[:,['CLOSE_BID','CLOSE_ASK']].mean(axis = 1)
    return(df[['idx','timestamp','close_price']])


def pctdiff_from_currprice (df, colname = 'close_price'):
    cols = df.columns.difference(['idx','timestamp',colname]).to_list()
    for col in cols:
        tmp = df[[colname,col]].pct_change(axis=1)
        df[col] = tmp[col]
    return df

############ EXPONENTIAL MOVING AVERAGES

# THESE ARE QUOTED IN BARS
def add_ema_custom (df, colname = 'close_price', window = 6, feature_name = "1min"):
    feature_name = "ema_" + feature_name
    df[feature_name] = df.loc[:,colname].ewm(span = window, adjust = False ).mean()
    return df


def add_many_emas (df, colname, window_vec, name_vec):
    for i in range(len(window_vec)):
        df = add_ema_custom(df, colname, window_vec[i], name_vec[i])
    return df


# only use future info for bogey construction
def add_future_ema (df, colname = 'close_price', window = 6, feature_name = '1min'):
    feature_name = 'ema_' + feature_name
    df[feature_name] = df.loc[:,colname][::-1].ewm(span=window, adjust=True)[::-1]
    return df


def add_many_future_emas (df, colname, window_vec, name_vec):
    for i in range(len(window_vec)):
        df = add_future_ema(df, colname, window_vec[i], name_vec[i])
    return df



############ SIMPLE MOVING AVERAGES

# THESE ARE QUOTED IN BARS
def add_sma_custom (df, colname = 'close_price', window = 6, feature_name = "1min"):
    feature_name = "sma_" + feature_name
    df[feature_name] = df.loc[:,colname].