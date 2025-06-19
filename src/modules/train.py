import  torch
from tqdm import tqdm
import numpy as np
import pandas as pd 
import random
from sklearn.metrics import r2_score
from .model_settings import *


from .config import device
from .figures import *
from .DQN_model import TradingNet
from .environment import TradingEnv
from .replay import ReplayBuffer
from .evaluate import evaluate

def train(train_data, test_data):
    #data = stock_data()
    env = TradingEnv(train_data,30)#30 is for days
    action_size = 3 # buy,sell, hold
    model = TradingNet().to(device) # neural network

    #network setup
    target_model = TradingNet().to(device)
    target_model.load_state_dict(model.state_dict())

  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    replay_buffer = ReplayBuffer(buffer_capacity)

    global epsilon # access epsilon variable from settings
    rewards_history = []
    net_worth_history = []
    qvalues_history =[]
    qtargets_history = []
    losses = []
    iteration_counter = []
    r2_history = []

    log_data = [] # empty list to store log values for later review 

    pbar = tqdm(range(episodes), desc="Training Episodes")

    _iteration = 0

    # Training Loop
    for episode in pbar:
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() <= epsilon:
                action = random.randrange(action_size)
            else:
                with torch.no_grad():
                    # convert to tensor and move to device.unsqueeze to make batch dimensions work.
                    action = torch.argmax(model(torch.FloatTensor(state).unsqueeze(0).to(device))).item()
            
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

             # iteration counter
            _iteration += 1


            # Replay
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                dones = torch.FloatTensor(dones).to(device)

                # values for model states
                q_values = model(states)
                next_q_values = target_model(next_states)
                # Q-values for next states
                max_next_q_values = torch.max(next_q_values, dim=1)[0]
                q_targets = rewards + gamma * max_next_q_values * (1 - dones)
                q_values_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze()




                loss = criterion(q_values_actions, q_targets)
                losses.append(loss)
                qvalues_history.append(q_values)
                qtargets_history.append(q_targets)
                iteration_counter.append(_iteration)

                # create progress bar for r2
                # Calculate R² for this iteration (per episode)
                r2_val = r2_score(q_targets.detach().cpu().numpy(), q_values_actions.detach().cpu().numpy())
                r2_history.append(r2_val)

                # log_data dictionary for iteration values
                log_data.append({"episode":episode,
                                 "iteration": _iteration,
                                 "q_values": q_values_actions,
                                 "q_targets": q_targets,
                                 "loss": loss.item(),
                                 "r2": r2_val,
                                 "net_worth": env.net_worth})
    

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            evaluate(model, test_data)
        target_model.load_state_dict(model.state_dict())
        net_worth_history.append(env.net_worth) # append the changes in network from the trading environement
        rewards_history.append(total_reward)


        # progress bar
        pbar.set_postfix({
        'NetWorth': f"{env.net_worth}",
        'Iteration': f"{_iteration}",
        'Epsilon': f"{epsilon:.3f}",
        'Loss': f"{loss.item():.4f}" if 'loss' in locals() else 'N/A',
        'R2' : f"{r2_val: .4f}" if 'r2_val' in locals() else 'N/A'
        })

      # Final R² calculation across all episodes
        all_qtargets = torch.cat(qtargets_history).detach().cpu().numpy()
        all_qvalues = torch.cat([q.gather(1, torch.argmax(q, dim=1, keepdim=True)).squeeze() for q in qvalues_history]).detach().cpu().numpy()
        final_r2 = r2_score(all_qtargets, all_qvalues)
        print(f"Final R² across training: {final_r2:.4f}")
    
    # save log_data values into dataframe
    df_log_data = pd.DataFrame(log_data)
    df_log_data.to_csv("../../output/log_data/log_data.csv")
    
    # save model
    torch.save(model.state_dict(),model_path)
    print("Model saved to", model_path)

    # display progress
    NetWorth_plot(net_worth_history)

    # plot R² history
    R_squared_plot(r2_history)

    # R² and Loss Plot
    R_squared_vs_Loss(r2_history, losses)


    # return trained model
    return model