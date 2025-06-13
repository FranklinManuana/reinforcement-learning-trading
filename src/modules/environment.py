

class TradingEnv:
    def __init__(self, df, window_size = 30, initial_balance = 10000):
        self.df = df.reset_index(drop=True)
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
        state = self.df.iloc[self.current_step - self.window_size:self.current_step].values
        return np.concatenate([state.flatten(), [self.balance, self.shares_held]])
    
    def step(self, action):
        price = self.df.iloc[self.current_step]['close']
        prev_net = self.net_worth

        if action == 1:
            if self.balance >= price:
                self.balance -= price
                self.shares_held += 1 
        elif action == 2: # sell
            if self.shares_held > 0:
                self.balance += price 
                self.shares_held -= 1 
    
    self.current_step += 1
    self.done = self.current_step >= len(self.df)
    self.net_worth = self.balance + self.shares_held * price
    reward = self.net_worth - prev_net

    return self.getStates(), reward, self.done