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
    df[feature_name] = df.loc[:,colname].