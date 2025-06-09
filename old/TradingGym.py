import gym
import pandas as pd
import numpy as np
import pickle
import math
from gym.spaces import Box, Dict, MultiDiscrete

class TradingGym(gym.Env):
    def __init__(self, api_client, window_size=10, initial_cash=10000):
        super(TradingGym, self).__init__()
        self._PROFIT_RATIO = 2
        self.api_client = api_client 
        self.window_size = window_size     # how many past prices the agent sees
        self.initial_cash = initial_cash
        self.current_step = 0

        with open('DUOL.pkl', 'rb') as file:
            self.data = pickle.load(file)

        # Box([0,3] x [0, 100]):
        # Coordinate 1: 0 = hold, 1 = buy, 2 = sell
        # Coodrinate 2: Number 
        self.action_space = MultiDiscrete([3, 30], dtype="int")
        
        # Observation: vector of length window_size (price history),
        # plus two scalars [cash, shares]
        obs_dim = window_size + 2
        
        # e.g. prices normalized between 0 and 1; cash scaled, shares scaled
        self.observation_space = Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # Internal state
        self.reset()
        

    def step(self, action: int):
        price = self.prices[self.current_step + self.window_size - 1]
        if action[0] == 1:
            self.shares_held += (action[1] if self.cash >= action[1] * price else math.floor(self.cash / price))
            self.cash = max(0, self.cash - action[1] * self.prices)

    
    def reset(self, seed=None, options=None):
        # Seed self.np_random
        super().reset(seed=seed)

        self.current_step = 0

        # Portfolio state
        self.cash = self.initial_cash
        self.shares_held = 0

        # Get window of prices
        self.prices = self.data['Close'].values

        # Construct initial observation
        price_window = np.array(self.prices[self.current_step:self.current_step + self.window_size])
        normalized_price_window = price_window / np.max(price_window)

        obs = np.concatenate(
            [
                normalized_price_window.astype(np.float32),
                self.cash / (self.initial_case * self._PROFIT_RATIO),
                self.shares_held / 100.0
            ]
        )

        return obs
    
    def render(self):
        return

