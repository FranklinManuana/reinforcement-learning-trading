from .environment import TradingEnv
import torch
from .config import device
# Evaluation
def evaluate(model, test_data, window_size=30):
    env = TradingEnv(test_data, window_size)
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            action = torch.argmax(model(torch.FloatTensor(state).unsqueeze(0).to(device))).item()
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
        networth = env.net_worth
    return networth