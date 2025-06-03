import gym
import pandas as pd
import numpy as np
from gym.spaces import Box, Dict, Discrete

class TradingGym(gym.Env):
    def __init__(self, api_client, window_size=10, initial_cash=10000):
        super(TradingGym, self).__init__()
        
        self.api_client = api_client 
        self.window_size = window_size     # how many past prices the agent sees
        self.initial_cash = initial_cash
        self.current_step = 0
        

        # Discrete(3): {0: hold, 1: buy, 2: sell}
        self.action_space = Discrete(3)
        
        # Observation: vector of length window_size (price history),
        # plus two scalars [cash, shares]
        obs_dim = window_size + 2
        # e.g. prices normalized between 0 and 1; cash scaled, shares scaled
        self.observation_space = Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # Internal state
        self.reset()
        

    def step(self):
        return
    
    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = 0
        self.portfolio_value = self.inital_value
        return
    
    def render(self):
        return

