def create_features(ticker,trading_window=21):
    #Function to calculate useful statistics from raw data
    #Trading window in working days
    
    import numpy as np
    import pandas as pd
    
    raw_data = pd.read_csv('Stocks/' + ticker + '.txt')
    raw_data.set_index('Date',inplace = True)
    
    features = pd.DataFrame(index=raw_data.index)
    
    features['Open'] = raw_data['Open']
    
    #Close std
    #features['std_q_tw'] = raw_data['Close'].shift(1).rolling(int((1/4)*trading_window)).std()
    #features['std_h_tw'] = raw_data['Close'].shift(1).rolling(int((1/2)*trading_window)).std()
    #features['std_tw'] = raw_data['Close'].shift(1).rolling(int((1)*trading_window)).std()
    #features['std_21_d'] = raw_data['Close'].shift(1).rolling(21).std()
    #features['std_tw'] = raw_data['Close'].shift(1).rolling(trading_window).std()
    #features['std_4tw'] = raw_data['Close'].shift(1).rolling(4*trading_window).std()
    
    #Overnight pct (Open_t - Close_t-1)
    features['pct_overnight'] = raw_data['Open'].div(raw_data['Close'].shift(1)) - 1
    
    #Close pct (Close - Close) & Close - Close std
    features['pct_close'] = raw_data['Close'].shift(1).div(raw_data['Close'].shift(2)) - 1
    
    features['pct_close_sigma_q_tw'] = features['pct_close'].div(features['pct_close'].rolling(int((1/4)*trading_window)).std())
    #features['pct_close_sigma_h_tw'] = features['pct_close'].div(features['pct_close'].rolling(int((1/2)*trading_window)).std())
    features['pct_close_sigma_tw'] = features['pct_close'].div(features['pct_close'].rolling(int((1)*trading_window)).std())                                                                         
    #features['pct_close_sigma_21'] = features['pct_close'].div(features['pct_close'].rolling(21).std())
    #features['pct_close_sigma_tw'] = features['pct_close'].div(features['pct_close'].rolling(trading_window).std())
    #features['pct_close_sigma_4tw'] = features['pct_close'].div(features['pct_close'].rolling(4*trading_window).std())
    
    #Average close return (Momentum)
    
    features['avg_return_q_tw'] = features['pct_close'].rolling(int((1/4)*trading_window)).mean()
    #features['avg_return_h_tw'] = features['pct_close'].rolling(int((1/2)*trading_window)).mean()
    features['avg_return_tw'] = features['pct_close'].rolling(int((1)*trading_window)).mean() 
                                                                                                   
    #features['avg_return_21'] = features['pct_close'].rolling(21).mean()
    #features['avg_return_tw'] = features['pct_close'].rolling(trading_window).mean()
    #features['avg_return_4tw'] = features['pct_close'].rolling(4*trading_window).mean()
    
    #Open percentile
    def rank(array):
        s = pd.Series(array)
        return s.rank(ascending=True,pct=True)[len(s)-1]
    
    features['perctentile_q_tw'] = features['Open'].rolling(int((1/4)*trading_window)).apply(rank,raw=False)
    #features['perctentile_h_tw'] = features['Open'].rolling(int((1/2)*trading_window)).apply(rank,raw=False)
    features['perctentile_tw'] = features['Open'].rolling(int((1)*trading_window)).apply(rank,raw=False)    
                                                   
    #features['perctentile_tw'] = features['Open'].rolling(trading_window).apply(rank,raw=False)
    #features['perctentile_4tw'] = features['Open'].rolling(4*trading_window).apply(rank,raw=False)
    
    #Volume
    features['pct_volume'] = raw_data['Volume'].shift(1).div(raw_data['Volume'].shift(2)) - 1
    
    features['pct_volume_sigma_q_tw'] = features['pct_volume'].div(features['pct_volume'].rolling(int((1/4)*trading_window)).std())
    #features['pct_volume_sigma_h_tw'] = features['pct_volume'].div(features['pct_volume'].rolling(int((1/2)*trading_window)).std())
    features['pct_volume_sigma_tw'] = features['pct_volume'].div(features['pct_volume'].rolling(int((1)*trading_window)).std())                                                                                              
                                                   
    #features['pct_volume_sigma_21'] = features['pct_volume'].div(features['pct_volume'].rolling(21).std())
    #features['pct_volume_sigma_tw'] = features['pct_volume'].div(features['pct_volume'].rolling(trading_window).std())
    #features['pct_volume_sigma_4tw'] = features['pct_volume'].div(features['pct_volume'].rolling(4*trading_window).std())
    
    features.drop('pct_volume',axis=1,inplace = True)
    
    #Remove rows with na features
    features.dropna(inplace=True)
    
    return features, raw_data