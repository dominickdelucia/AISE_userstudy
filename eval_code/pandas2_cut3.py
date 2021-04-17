import pandas as pd
import numpy as np
import itertools
import helpers
import preprocess


def closeprice_simplify_price_df(price_df):
    df = price_df.iloc[:, ~price_df.columns.str.match('b')]
    df['idx'] = df.index
    df['timestamp'] = df.TS_END
    df['close_price'] = df.loc[:, ['CLOSE_BID', 'CLOSE_ASK']].mean(axis=1)
    return (df[['idx', 'timestamp', 'close_price']])


def pctdiff_from_currprice(df, colname='close_price'):
    cols = df.columns.difference(['idx', 'timestamp', colname]).to_list()
    for col in cols:
        tmp = df[[colname, col]].pct_change(axis=1)
        df[col] = tmp[col]
    return df


############ EXPONENTIAL MOVING AVERAGES

# THESE ARE QUOTED IN BARS
def add_ema_custom(df, colname='close_price', window=6, feature_name="1min"):
    feature_name = "ema_" + feature_name
    df[feature_name] = df.loc[:, colname].ewm(span=window, adjust=False).mean()
    return df


def add_many_emas(df, colname, window_vec, name_vec):
    for i in range(len(window_vec)):
        df = add_ema_custom(df, colname, window_vec[i], name_vec[i])
    return df


# only use future info for bogey construction
def add_future_ema(df, colname='close_price', window=6, feature_name='1min'):
    feature_name = 'ema_' + feature_name
    df[feature_name] = df.loc[:, colname][::-1].ewm(span=window, adjust=True)[::-1]
    return df


def add_many_future_emas(df, colname, window_vec, name_vec):
    for i in range(len(window_vec)):
        df = add_future_ema(df, colname, window_vec[i], name_vec[i])
    return df


############ SIMPLE MOVING AVERAGES

# THESE ARE QUOTED IN BARS
def add_sma_custom(df, colname='close_price', window=6, feature_name="1min"):
    feature_name = "sma_" + feature_name
    df[feature_name] = df.loc[:, colname].rolling(window=window).mean()
    return df


def add_many_smas(df, colname, window_vec, name_vec):
    for i in range(len(window_vec)):
        df = add_sma_custom(df, colname, window_vec[i], name_vec[i])
    return df


# only use future info for bogey construction
def add_future_sma(df, colname='close_price', window=6, feature_name="1min"):
    feature_name = "sma_" + feature_name
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
    df[feature_name] = df.loc[:, colname].rolling(window=indexer).mean()
    return df


def add_many_future_smas(df, colname, window_vec, name_vec):
    for i in range(len(window_vec)):
        df = add_future_sma(df, colname, window_vec[i], name_vec[i])
    return df


######## MAKING INTERACTIONS

def make_squ_interactions(feature_df):
    for col in feature_df.columns:
        feature_df[col + '_sq'] = feature_df[col] ** 2

    ## MAKING INTERACTIONS
    for x in itertools.combinations(feature_df.columns, 2):
        feature_df[x[0] + "X" + x[1]] = feature_df[x[0]] * feature_df[x[1]]

    return feature_df


def make_polynomial_features(feature_df):
    all_cols = helpers.get_alphas_df(feature_df).columns
    for col in all_cols:
        feature_df[col + '_squ'] = feature_df[col] ** 2
        feature_df[col + '_cub'] = feature_df[col] ** 3
    return feature_df


def make_all_combin_interactions(feature_df):
    all_cols = helpers.get_alphas_df(feature_df).columns
    for x in itertools.combinations(all_cols, 2):
        feature_df[x[0] + "X" + x[1]] = feature_df[x[0]] * feature_df[x[1]]
    return feature_df


def make_int_and_poly_feats(feature_df):
    feature_df = make_all_combin_interactions(feature_df)
    feature_df = make_polynomial_features(feature_df)
    return feature_df


'''
Volume Features - trailing average volume, current bar volume, and diff between buy/sell volume, average buy/sell volume diff

Volatility Features - give features of trailing average interbar spread, recent bar spread, trailing price vol

Order Book Features - size of order book, spread on buy and sell sides

Time of day Features - give time of day (NY est time) 

'''


######## Making Volume Features

def grab_volume_features(price_df):
    df = price_df.iloc[:, ~price_df.columns.str.match('b')]
    df['idx'] = df.index
    df['timestamp'] = df.TS_END
    df['close_price'] = df.loc[:, ['CLOSE_BID', 'CLOSE_ASK']].mean(axis=1)

    df['total_volume'] = df.loc[:, ['VOL_BUY', 'VOL_SELL']].sum(axis=1)
    df['vol_buy'] = df.loc[:, ['VOL_BUY']]
    df['vol_sell'] = df.loc[:, ['VOL_SELL']]

    df['near_spread'] = df.apply(lambda x: x.ASK0_PRICE - x.BID0_PRICE, axis=1)
    df['far_spread'] = df.apply(lambda x: x.ASK4_PRICE - x.BID4_PRICE, axis=1)

    df['bid_volume'] = df.loc[:, ['BID0_QTTY', 'BID1_QTTY', 'BID2_QTTY', 'BID3_QTTY', 'BID4_QTTY']].sum(axis=1)
    df['ask_volume'] = df.loc[:, ['ASK0_QTTY', 'ASK1_QTTY', 'ASK2_QTTY', 'ASK3_QTTY', 'ASK4_QTTY']].sum(axis=1)

    df['bid_VWA_diff'] = df.apply(lambda x: (
                                                        x.BID0_PRICE * x.BID0_QTTY + x.BID1_PRICE * x.BID1_QTTY + x.BID2_PRICE * x.BID2_QTTY + x.BID3_PRICE * x.BID3_QTTY + x.BID4_PRICE * x.BID4_QTTY) / x.bid_volume - x.close_price,
                                  axis=1)
    df['ask_VWA_diff'] = df.apply(lambda x: (
                                                        x.ASK0_PRICE * x.ASK0_QTTY + x.ASK1_PRICE * x.ASK1_QTTY + x.ASK2_PRICE * x.ASK2_QTTY + x.ASK3_PRICE * x.ASK3_QTTY + x.ASK4_PRICE * x.ASK4_QTTY) / x.ask_volume - x.close_price,
                                  axis=1)

    return_cols = ['idx', 'timestamp', 'close_price', 'total_volume', 'vol_buy', 'vol_sell', 'near_spread',
                   'far_spread',
                   'bid_volume', 'ask_volume', 'bid_VWA_diff', 'ask_VWA_diff']

    return df.loc[:, return_cols]


def augment_volume_features(vol_df, roll_bars):
    df = vol_df.loc[:, ['idx', 'timestamp', 'close_price']]
    # need to take rolling Z score 1xfwd, 3xfwd, 6xfwd
    z_feats = ['close_price', 'total_volume', 'vol_buy', 'vol_sell', 'near_spread', 'far_spread', 'bid_volume',
               'ask_volume', 'bid_VWA_diff', 'ask_VWA_diff']
    for col in z_feats:
        df[col + '_' + str(roll_bars) + '_rollZ'] = preprocess.rolling_zscore(vol_df, col, roll_bars)
        df[col + '_' + str(3 * roll_bars) + '_rollZ'] = preprocess.rolling_zscore(vol_df, col, 3 * roll_bars)
        df[col + '_' + str(6 * roll_bars) + '_rollZ'] = preprocess.rolling_zscore(vol_df, col, 6 * roll_bars)

    # need to take VWA bid & ask diff to WAP if WAP is a reasonable measure
    return df


def vol_feat_generation(raw_df, roll_bars=360):
    target = 'close_price'
    xx = grab_volume_features(raw_df)
    df = augment_volume_features(xx, roll_bars)
    cols = df.columns.difference(['idx', 'timestamp', target])
    for col in cols:
        df = df.rename(columns={col: "{}.Alpha".format(col)})
    return (df)


def fg_ema_reversion_augmentation(raw_df, window_vec, name_vec):
    target = 'close_price'
    df = closeprice_simplify_price_df(raw_df)
    # making the EMA's needed

    df = add_many_emas(df, target, window_vec, name_vec)

    df = pctdiff_from_currprice(df)

    # name the columns for identifying "alpha" features
    cols = df.columns.difference(['idx', 'timestamp', target])
    for col in cols:
        df = df.rename(columns={col: "{}.Alpha".format(col)})

    #     df = fg.make_int_and_poly_feats(df)
    return (df)


####################### Bar Counters

def bar_streak(df, bar_lookback=30):
    # Assumes raw has already been simplified
    # df = simplify_price_df(raw_df)
    cols = list(df.columns)
    df['pos_bar'] = (df['close_price'] > df['open_price']) * 1
    df['neg_bar'] = (df['open_price'] > df['close_price']) * 1
    tmp_p = pd.DataFrame({'0': df.pos_bar})
    tmp_n = pd.DataFrame({'0': df.neg_bar})
    for i in range(1, bar_lookback):
        tmp_p[str(i)] = df.pos_bar.rolling(window=i, min_periods=1).sum() * df.pos_bar.shift(i)
        tmp_n[str(i)] = df.neg_bar.rolling(window=i, min_periods=1).sum() * df.neg_bar.shift(i)
    tmp_p['streak'] = tmp_p.