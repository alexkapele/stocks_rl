import gym
from gym import error, spaces, utils
from gym.utils import seeding
from create_features import create_features
import numpy as np
import pandas as pd

class MarketEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, pos_limit=2, trading_window=21, trading_freq=1, stop_loss_thres=0.98, take_prof_thres=1.05, ticker=None, features=None, raw_data=None):
              
        kwargs = {'pos_limit': pos_limit,
                  'trading_window': trading_window,
                  'trading_freq':trading_freq,
                  'stop_loss_thres': stop_loss_thres,
                  'take_prof_thres': take_prof_thres,
                  'ticker': ticker,
                  'features':features,
                  'raw_data':raw_data}
        
        self.pos_limit = pos_limit
        self.trading_window = trading_window
        self.trading_freq = trading_freq
        self.stop_loss_thres = stop_loss_thres
        self.take_prof_thres = take_prof_thres
        self.ticker = ticker
        
        #calculate all features and prices for entire environment
        self.features_all = features
        self.prices_all = raw_data
                
        self.action_space = spaces.Discrete(3) #hold, buy, sell
        self.action_encoding = {0: 0, 1: 1, 2: -1}
    
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.features_all.shape[1],0), dtype=np.float32)
        #n features: signals + position (int) + abs(pos)-abs(pos_limit) + time (int)

        self.seed()
    
    def seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]    
    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        #Get global variables
        trade_log = self.trade_log
        pos_limit = self.pos_limit
        action_log = self.action_log
        pos_log = self.pos_log
        
        #Get state and action
        state = self.state
        features_t,pos,pos_diff,time_to_complete,stop_loss,take_profit = state
        action = self.action_encoding[action] #convert action from 0,1,2 to 0,1,-1
        
        self.stop_loss_log.append(self.stop_loss) #Save stop_loss log for each step

        lockin_pl = 0 #reset pl
        
        #Take action (depending on pos_limit & stop_loss/take_profit
        if action*pos>=0 and self.stop_loss==False and self.take_profit==False:
            if abs(pos) < pos_limit:
                trade_log.append(action)
                pos += action #increase existing position or apply action if flat or hold
            else:
                trade_log.append(0) 
            self.hold_time += abs(pos) #increase hold time for all active positions
        else:
            #trade_log.append(action*(abs(pos)+0))
            trade_log.append(-pos)
            pos = 0 #action #close all existing positions and apply current action          
            self.hold_time = abs(pos) #reset hold time
            lockin_pl = self.active_return #active return up to date when position is closed (for reward)
        
        #Save logs
        action_log.append(action)
        pos_log.append(pos)  
        self.action_log = action_log
        self.pos_log = pos_log
        self.trade_log = trade_log
        
        #Calculate inactive time for pernalty (i.e. no position)
        if action == 0 and pos == 0:
            self.inactive_time += 1
        else:
            self.inactive_time = 0         
        
        #Store day when last position was openned (for stop_loss & take_profit calc)
        if (pos_log[self.window_counter]!=0) and (pos_log[self.window_counter-1]==0):        
            self.day_pos = self.window_counter
        
        
        #Check if episode is finished
        done = True if self.window_counter == self.trading_window-1 else False
        
       
        #Calculate returns up to today's close
        close_open_returns = self.prices.iloc[-1]['Close'] / self.prices.iloc[:self.window_counter+1]['Open']
        close_returns = [1+trade_log[i]*(close_open_returns[i]-1) for i in range(len(close_open_returns))]       
        total_return = np.prod(close_returns)
        
        #Store returns up to each day's close
        self.daily_returns.append(total_return)
        #sharpe = total_return/np.std(self.daily_returns)
        
        
        #Calculate next day's open over all previous days' open (for stop_loss & take_profit trigger)
        if self.window_counter < self.trading_window-1:
            open_open_return = self.prices.iloc[self.window_counter+1]['Open'] / self.prices.iloc[:self.window_counter+1]['Open']
        else:
            open_open_return = self.prices.iloc[self.window_counter]['Close'] / self.prices.iloc[:self.window_counter+1]['Open']
        #Calculate active return (marked at Open_t+1)
        if pos != 0:
            self.today_return = [1+trade_log[i]*(open_open_return[i]-1) for i in range(self.day_pos,self.window_counter+1)]
            self.active_return = np.prod(self.today_return)
        else:
            self.active_return = 1   
        
        
        #Check stop_loss and take_profit
        self.stop_loss = True if self.active_return < self.stop_loss_thres else False
        self.take_profit = True if self.active_return > self.take_prof_thres else False
        
         #Set next step's state
        if done==False:
            self.state = (self.features.iloc[self.window_counter+1,:], pos, abs(pos)-abs(pos_limit), self.trading_window-self.window_counter, self.stop_loss, self.take_profit)
        
        
        #Calculate average return of the following days (for reward function)
        avg_return = (self.prices.iloc[self.window_counter:]['Close']).mean()/self.prices.iloc[self.window_counter]['Open']
        
        #Reward function weights
        c_return = 1
        c_daily_return = 0.85
        c_hold = -0.02
        c_trades = -0.03
        c_inactive = -0.15
        c_lockin = 0.5
        
        if close_returns[-1]<1:
            reward_ = -3
        elif close_returns[-1]>1:
            reward_ = 1
        else: 
            reward_ = 0

        reward = c_return*reward_ + c_daily_return*np.sign(avg_return-1) + c_hold*self.hold_time + c_trades*sum(np.absolute(trade_log)) + c_inactive*self.inactive_time +  c_lockin*np.sign(lockin_pl-1)
        
        
        self.window_counter += self.trading_freq #Update window_counter
        
        return np.array(self.state[0].tolist()+[self.state[1], self.state[2]]), reward, done, {'trades': self.trade_log, 'actions': self.action_log, 'pos': self.pos_log, 'prices': self.prices, 'total_return': total_return, 'close_return': close_returns, 'close_open_returns': close_open_returns, 'daily_return': close_returns[-1], 'stop_loss_log': self.stop_loss_log}
        
    def reset(self):
        
        #sample episode's signals
        i = np.random.randint(0,high=len(self.features_all)-self.trading_window-1)
        self.features = self.features_all.iloc[i:i+self.trading_window,]
        self.prices = self.prices_all.loc[self.features.index,]
        
        self.window_counter = 0
        self.inactive_time = 0
        self.hold_time = 0
        
        self.active_return = 1
        self.day_pos = 0
        self.stop_loss = False
        self.take_profit = False
                
        self.trade_log = [] #to keep log of all trades
        self.action_log = []
        self.pos_log = []
        self.stop_loss_log = []
        
        self.today_return = []
        self.daily_returns = []
        
        self.state = (self.features.iloc[0,:],0,0,self.trading_window, self.stop_loss, self.take_profit)
        #self.state = (self.features.iloc[0,:],0,0)
        
        
        return np.array(self.state[0].tolist()+[self.state[1], self.state[2]])
    
        
        
