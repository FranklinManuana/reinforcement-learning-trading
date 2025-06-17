import numpy as np


class TradingEnv:
    def __init__(self, data, window_size = 30, initial_balance = 10000):
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.done = False
        return self.getStates()
    
    def getStates(self):
        state = self.data[self.current_step - self.window_size:self.current_step].flatten()
        # normalize balance and shares_held before 
        norm_balance = self.balance / self.initial_balance
        norm_shares = self.shares_held
        norm_net_worth = self.net_worth/self.initial_balance
        return np.concatenate((state,[norm_balance, norm_shares,norm_net_worth]))
    
    def step(self, action):
        price = self.data[self.current_step][3] # close
        prev_net = self.net_worth
        reward = 0

        if action == 1: # Buy
            if self.balance >= price:
                self.balance -= price
                self.shares_held += 1 
        elif action == 2: # sell
            if self.shares_held > 0:
                self.balance += price 
                self.shares_held -= 1
    
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.done = True
        self.net_worth = self.balance + self.shares_held * price
        reward = self.net_worth - prev_net

        return self.getStates(), reward, self.done