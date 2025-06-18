# Reinforcement Learning Trading 📈

This project implements a Deep Q-Learning (DQN) agent for financial trading using a custom reinforcement learning environment. It is designed to simulate and train a trading agent using historical financial data, neural networks, and experience replay.

---

## 🔍 Overview

This project offers:

- 🔄 A modular design for ease of use and repurposing
- Integration with real and historical market data sources
- Utilities for plotting, evaluation, logging, and analysis

---

## 📂 Repository Structure

```
reinforcement-learning-trading/
│
├── output/                          # Output files generated during training/evaluation
│   ├── figures/                     # Plots and visualizations
│   └── trained_models/             # Saved model weights/checkpoints
│
├── src/                             # Core source code
│   ├── modules/                     # Modular Python files
│   │   ├── config.py               # Configuration and seed setup
│   │   ├── DQN_model.py           # Deep Q-Network (DQN) model architecture
│   │   ├── environment.py         # Trading environment logic
│   │   ├── evaluate.py            # Evaluation logic for trained agents
│   │   ├── figures.py             # Plotting utilities (e.g., loss, R²)
│   │   ├── model_settings.py      # Hyperparameters and training setup
│   │   ├── replay.py              # Experience replay buffer implementation
│   │   ├── stock_data.py          # Stock data loading and preprocessing
│   │   └── train.py               # Model training loop
│
├── notes/                           # (Optional) Notes or experiment logs
│
├── main.ipynb                       # Entry-point Jupyter notebook for running the model
├── README.md                        # Project overview and documentation
```
---

## Phases

### 1. Trading Environment
A custom trading environment built to simulate portfolio management tasks. It tracks:
- Balance
- Shares held
- Net worth
- Actions: buy, hold, sell

### 2. Network Architecture
A deep neural network (DQN variant) is constructed using PyTorch. The architecture includes:
- Fully connected layers
- LeakyReLU activations
- Q-value outputs for trading actions

### 3. Replay Buffer
An experience replay class that stores past transitions (state, action, reward, next state) to sample mini-batches during training. This helps stabilize and decorrelate learning.

### 4. Parameters / Settings
Hyperparameters include:
- Learning rate
- Discount factor (gamma)
- Epsilon-greedy parameters
- Batch size
- Episode length

### 5. Data Loading
Historical price data is loaded and preprocessed to be used in the environment. Supports:
- Multiple tickers
- Train/test splits (e.g., by year)

### 6. Training
The DQN agent is trained over multiple episodes with:
- Epsilon-greedy exploration
- Gradient updates using MSE loss
- Target Q-value calculation

### 7. Evaluation
Post-training evaluation includes:
- Performance visualization (net worth over time)
- R² and loss logging per iteration
---

## 🎯 Features

- **Modular environments**: Swap between stock, crypto, or multi-asset paper trading setups.
- **Multiple RL algorithms**: DQN, Double DQN, PPO, SAC, etc. — easily extendable.
- **Metrics & plots**: Analyze returns, Sharpe, drawdowns, and compare against baselines.
- **Real-time trading**: Paper/live trading mode for demo or backtesting.

---

## 🔧 Configuration

Edit `model_settings.py`  `config.py` to adjust settings such as:
`model_settings.py`
- Data source / preprocessing rules
- Environment parameters (e.g., window size, action space)
- RL agent hyperparameters (learning rate, gamma, etc.)
- Logging & checkpoint directories
`config.py`

---

## 📊 Logs & Output

- plots, metrics, and models are saved to output📂 
- `figures.py` use to visually display:
  - Training loss & episode returns
  - Portfolio Networth
---

---

## ✅ Requirements

- Python 3.9+
- Core libraries: `torch`, `numpy`, `pandas`, `matplotlib`
- RL frameworks: optional

---

## 📄 Best Practices

- 🧠 Use **non-leaky feature windows** to avoid lookahead bias
- 💸 Include real-world factors: transaction costs, slippage, and order fill assumptions
- 🗕️ **Train/test split** should be chronological to mimic real deployment
- 📍 Maintain **random seed control** for reproducible results

---

## 📚 References

   📚
- [yFinance Documentation](https://ranaroussi.github.io/yfinance/)
- [TensorFlow Introduction to DQN](https://www.tensorflow.org/agents/tutorials/0_intro_rl)
- [Medium Article on Deep Q-Learning (DQN)](https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae) by Samina Amin 2024
- [Medium Article on Deep Q-Network(DQN)](https://medium.com/@shruti.dhumne/deep-q-network-dqn-90e1a8799871) by Shruti Dhumme

   
   💻
- [Making Neural Network From Scratch](https://www.youtube.com/watch?v=w8yWXqWQYmU&t=1325s)

---

## 🛠 Contribution & License

You're welcome to open issues, suggest algorithms, or add new environments. Pull requests are highly appreciated!

**License**: [Your Choice - MIT / Apache 2.0 / etc.]

---

### 🚀 What's next?
- 💸 Include real-world factors: transaction costs, slippage, and order fill assumptions
- Add risk-sensitive agents like **SAC** or **PPO**
- Integrate **live data APIs** (e.g. Alpaca, Binance)
- Add **multi-asset portfolio optimization**
