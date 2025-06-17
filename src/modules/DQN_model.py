import torch


N_features = 5 # OHLCV
window_size = 30 # 30-day trading window
N_states = N_features * window_size
N_states = N_states + 3 # +3 for balance, shares_held, networth

class TradingNet(torch.nn.Module):
    def __init__(self):
        super(TradingNet, self).__init__()
        self.fc1 = torch.nn.Linear(N_states,8192)
        self.fc2 = torch.nn.Linear(8192,1024)
        self.fc3 = torch.nn.Linear(1024,3)
        self.activ = torch.nn.LeakyReLU()
    
    def forward(self, x):
        x =self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        return self.fc3(x)