def threshold_revert_long_bogey(df, target, fwd_pred, prof_thresh, bogey_name = 'bogey'):
    fwd_bars = int(fwd_pred/BARSEC)
    tmp = pd.DataFrame({'idx': df.index, 'curr' : df[target].values})
    tmp.set_index('idx',drop = True, inplace = True)
    for i in range(fwd_bars):
        tmp[i] = tmp.curr.pct_change(i).shift(-i)
    tmp.drop('curr', axis = 'columns', inplace = True)
    tmp = tmp.applymap(lambda x:  1 if x > prof_thresh else 0)
    rev = pd.DataFrame({'idx': df.index, 'curr' : df[target].values})
    rev.set_index('idx', drop = True, inplace = True)
    window_vec = [fwd_bars,fwd_bars*2,int(fwd_bars/2)]
    name_vec = ['sma1','sma2','sma3']
    rev = fg.add_many_smas(rev, 'curr', window_vec, name_vec)
    rev = rev.apply(above_curr,  axis=1, result_type='expand')
    df['rev'] = np.multiply(rev.sum(axis=1) >= 2, 1)
    df['bogsum'] = tmp.