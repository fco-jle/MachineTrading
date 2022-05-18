import numpy as np

def standard_dev(df):
    close_ = df['Close'].values
    return np.std(close_)

def close_to_close(df):
    close_ = df['Close'].values
    returns = close_[1:] / close_[:-1]
    log_returns = np.log(returns)
    ctc = np.sqrt(np.mean(log_returns**2))
    return ctc

def parkinson(df):
    open_ = df['Open'].values
    high_ = df['High'].values
    low_ = df['Low'].values
    close_ = df['Close'].values
    pk = np.sqrt(1 / (4*np.log(2)) * np.mean(np.log(high_/low_)**2))
    return pk

def garman_klass(df):
    open_ = df['Open'].values
    high_ = df['High'].values
    low_ = df['Low'].values
    close_ = df['Close'].values

    f1 = np.mean(0.5*np.log(high_/low_)**2)
    f2 = np.mean((2*np.log(2)-1)*np.log(close_/open_)**2)
    gk = np.sqrt(f1-f2)
    return gk

def rogers_satchell(df):
    open_ = df['Open'].values
    high_ = df['High'].values
    low_ = df['Low'].values
    close_ = df['Close'].values
    vol = np.sqrt(np.mean(np.log(high_/close_)*np.log(high_/open_)+np.log(low_/close_)*np.log(low_/open_)))
    return vol

def garman_klass_yang_zhang(df):
    open_ = df['Open'].values[1:]
    high_ = df['High'].values[1:]
    low_ = df['Low'].values[1:]
    close_ = df['Close'].values[1:]
    prev_close = df['Close'].shift().values[1:]

    f1 = np.mean(np.log(open_/prev_close)**2)
    f2 = np.mean(0.5*np.log(high_/low_)**2)
    f3 = np.mean((2*np.log(2)-1)*(np.log(close_/open_)**2))
    vola = np.sqrt(f1+f2-f3)
    return vola

def yang_zhang(df):
    open_ = df['Open'].values[1:]
    high_ = df['High'].values[1:]
    low_ = df['Low'].values[1:]
    close_ = df['Close'].values[1:]
    prev_close = df['Close'].shift().values[1:]

    alpha = 1.34  # Suggested value in original Yang Zhang paper
    T = len(open_)
    k = (alpha-1) / (alpha + (T+1)/(T-1))

    rs = rogers_satchell(df)
    overnight = np.sum(np.power(np.log(open_/prev_close)-np.mean(np.log(open_/prev_close)), 2)) / (T-1)
    open_close = np.sum(np.power(np.log(close_/open_)-np.mean(np.log(close_/open_)), 2)) / (T-1)
    yz = np.sqrt( overnight + k*open_close + (1-k)*np.power(rs, 2))
    return yz
