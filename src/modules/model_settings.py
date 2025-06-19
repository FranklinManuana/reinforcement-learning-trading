from .config import *
# model parameter settings
episodes = 50
gamma = 0.9999
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999
batch_size = 128
buffer_capacity = 10000
model_path = model_path_name(episodes,TRAINED_DIR)