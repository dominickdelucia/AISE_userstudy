def threshold_revert_long_bogey(df, target, fwd_pred, prof_thresh, bogey_name='bogey'):
    fwd_bars = int(fwd_pred / BARSEC)
    tmp = pd.DataFrame({'idx': df.index, 'curr': df[target].values})
    tmp.set_index('idx', drop=True, inplace=True)
    for i in range(fwd_bars):
        tmp[i] = tmp.curr.pct_change(i).shift(-i)
    tmp.drop('curr', axis='columns', inplace=True)
    tmp = tmp.applymap(lambda x: 1 if x > prof_thresh else 0)
    rev = pd.DataFrame({'idx': df.index, 'curr': df[target].values})
    rev.set_index('idx', drop=True, inplace=True)
    window_vec = [fwd_bars, fwd_bars * 2, int(fwd_bars / 2)]
    name_vec = ['sma1', 'sma2', 'sma3']
    rev = fg.add_many_smas(rev, 'curr', window_vec, name_vec)
    rev = rev.apply(above_curr, axis=1, result_type='expand')
    df['rev'] = np.multiply(rev.sum(axis=1) >= 2, 1)
    df['bogsum'] = tmp.sum(axis=1)
    df[bogey_name] = (df['bogsum'] > 0) * df['rev']
    return (df)


def threshold_revert_short_bogey(df, target, fwd_pred, prof_thresh, bogey_name='bogey'):
    fwd_bars = int(fwd_pred / BARSEC)
    tmp = pd.DataFrame({'idx': df.index, 'curr': df[target].values})
    tmp.set_index('idx', drop=True, inplace=True)
    # getting percent change
    for i in range(fwd_bars):
        tmp[i] = tmp.curr.pct_change(i).shift(-i) * -1
    tmp.drop('curr', axis='columns', inplace=True)
    tmp = tmp.applymap(lambda x: 1 if x > prof_thresh else 0)

    rev = pd.DataFrame({'idx': df.index, 'curr': df[target].values})
    rev.set_index('idx', drop=True, inplace=True)

    window_vec = [fwd_bars, fwd_bars * 2, int(fwd_bars / 2)]  # these are in bars
    name_vec = ['sma1', 'sma2', 'sma3']
    rev = fg.add_many_future_smas(rev, 'curr', window_vec, name_vec)

    rev = rev.apply(below_curr, axis=1, result_type='expand')

    df['rev'] = np.multiply(rev.sum(axis=1) >= 2, 1)

    df['bogsum'] = tmp.sum(axis=1)
    df[bogey_name] = (df['bogsum'] > 0) * df['rev'] * -1

    return (df)


def above_curr(row, comp_col='curr'):
    tmp = []
    for i in range(1, len(row)):
        tmp.append(row[i] >