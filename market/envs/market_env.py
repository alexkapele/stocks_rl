import gym
from gym import error, spaces, utils
from gym.utils import seeding
from create_features import create_features
import numpy as np

class MarketEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.pos_limit = None #position limit (input)
        self.trading_window = None #trading window in days (input)
        self.trading_freq = None #
        
        self.ticker = None #ticker (input)
        self.features,self.prices = create_features(ticker,trading_window,True)
        
        self.action_space = Spaces.Discrete(3) #hold, buy, sell
    
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(19,0), dtype=np.float32)
        #n features: 17 signals + position (int) + time (int)
    
        self.window_counter = 0
        
        self.trade_log = [] #to keep log of all trades
        self.daily_returns = [] 
    
        self.seed()
    
    def seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]    
    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        features_t[0:15],pos,hold_time = state
        
        if action*pos>=0:
            trade_log.append(action)
            pos += action #increase existing position or apply action if flat or hold
            pos = min(pos,pos_limit) if pos >= 0 else max(pos,-pos_limit) 
            hold_time += abs(pos) #increase hold time for all active positions
        elif action*pos<0:
            trade_log.append(action*abs(pos+1))
            pos = action #close all existing positions and apply current action          
            hold_time = abs(pos) #reset hold time
                           
        self.state = (features[window_counter+1,:],pos,hold_time)
        
        done = True if window_counter == trading_window-1 else False
        
        ''' Reward function '''
        #calculate returns up to today's close
        close_open_returns = prices.iloc[window_counter,'Close'] / prices.iloc[:window_counter,'Open']
        close_returns = abs(trade_log[i]) - trade_log[i]*(1-close_open_returns[i]) for i in range(window_counter)
        total_return = np.prod(daily_returns)
        
        daily_returns.append(total_return)
        sharpe = total_return/np.std(daily_returns)
        
        a=1
        b=0.005
        c=0.05
        reward = a*sharpe + b*hold_time + c*sum(np.absolute(trade_log))
        
        
        window_counter += 1
        
        return np.array(self.state), reward, done, {}
        
    def reset(self):
        self.state = (features[0,:],0,0)
    return np.array(self.state)
    
    def render(self, mode='human', close=False):
