import gym
from gym import error, spaces, utils
from gym.utils import seeding
from create_features import create_features
import numpy as np
import pandas as pd

class MarketEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, pos_limit=2, trading_window=21, trading_freq=1, ticker=None):
        #self.pos_limit = None #position limit (input)
        #self.trading_window = None #trading window in days (input)
        #self.trading_freq = None #
        #self.ticker = None #ticker (input)
        
        kwargs = {'pos_limit': pos_limit,
                  'trading_window': trading_window,
                  'trading_freq':trading_freq,
                  'ticker': ticker}
        #super().__init__(**kwargs)
        
        self.pos_limit = pos_limit
        self.trading_window = trading_window
        self.trading_freq = trading_freq
        self.ticker = ticker
        
        #calculate all features and prices for entire environment
        self.features_all,self.prices_all = create_features(self.ticker,self.trading_window,False)
                
        self.action_space = spaces.Discrete(3) #hold, buy, sell
        self.action_encoding = {0: 0, 1: 1, 2: -1}
    
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(19,0), dtype=np.float32)
        #n features: 17 signals + position (int) + time (int)

        self.seed()
    
    def seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]    
    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        trade_log = self.trade_log
        pos_limit = self.pos_limit
        
        action_log = self.action_log
        pos_log = self.pos_log
               
        state = self.state
        action = self.action_encoding[action]
        
        features_t,pos,hold_time = state
        
        if action*pos>=0:
            trade_log.append(action)
            pos += action #increase existing position or apply action if flat or hold
            pos = min(pos,pos_limit) if pos >= 0 else max(pos,-pos_limit) 
            hold_time += abs(pos) #increase hold time for all active positions
        elif action*pos<0:
            trade_log.append(action*abs(pos)+1)
            pos = action #close all existing positions and apply current action          
            hold_time = abs(pos) #reset hold time
        
        action_log.append(action)
        pos_log.append(pos)  
        self.action_log = action_log
        self.pos_log = pos_log
        
        self.trade_log = trade_log
        
        done = True if self.window_counter == self.trading_window-1 else False
        
        if done==False:
            self.state = (self.features.iloc[self.window_counter+1,:], pos, hold_time)
        
        ''' Reward function '''
        #calculate returns up to today's close
        close_open_returns = self.prices.iloc[self.window_counter]['Close'] / self.prices.iloc[:self.window_counter]['Open']
        #close_returns = [abs(trade_log[i]) - trade_log[i]*(1-close_open_returns[i]) for i in range(self.window_counter)]
        close_returns = [1+trade_log[i]*(close_open_returns[i]-1) for i in range(self.window_counter)]
        total_return = np.prod(close_returns)
        
        self.daily_returns.append(total_return)
        #sharpe = total_return/np.std(self.daily_returns)
        
        a=1
        b=0.005
        c=0.05
        reward = a*total_return + b*hold_time + c*sum(np.absolute(trade_log))
        
        
        self.window_counter += self.trading_freq
        
        return np.array(self.state[0].tolist()+[self.state[1], self.state[2]]), reward, done, total_return, {}
        
    def reset(self):
        
        #sample episode's signals
        i = np.random.randint(0,high=len(self.features_all)-self.trading_window-1)
        self.features = self.features_all.iloc[i:i+self.trading_window,]
        self.prices= self.prices_all.loc[self.features.index,]
        
        self.window_counter = 0
        
        self.trade_log = [] #to keep log of all trades
        self.action_log = []
        self.pos_log = []
         
        self.daily_returns = []
        
        self.state = (self.features.iloc[0,:],0,0)
        
        return np.array(self.state)
    
    #def render(self, mode='human', close=False):
        
        
