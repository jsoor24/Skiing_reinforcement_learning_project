from Env import Env
from RandomAgent import RandomAgent
from DQNAgent import DQNAgent
import matplotlib.pyplot as plt
import os
import torch 

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def play():
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
    environment.playEnvironment(mapping)

def plotLearningGraphs(learning_stats):
    print("Plotting graphs")
    for idx in range(len(learning_stats)):
        plt.plot(learning_stats[idx])
        if(idx==0):
            ylabel="Loss difference"
        elif(idx==1):
            ylabel="Total reward"
        elif(idx==2):
            ylabel="Episode length"
        elif(idx==3):
            ylabel="Epsilon"
        plt.ylabel(ylabel)
        plt.show()
    return

# Initialise skiing environment.
env = Env('Skiing-v4')

# Initialise agent using environment.
agent = RandomAgent(env)

# Create DQN agent.
dqn_agent = DQNAgent(env=env, learning_rate=1e-3, sync_freq=5, replay_buffer_size=256)

# Train agent.
learning_stats = dqn_agent.train(2)
print(learning_stats)
print("Saving trained model")
dqn_agent.save_trained_model("cartpole-dqn.pth")

# Load the agent mode
#dqn_agent.load_pretrained_model("optimal-policy.pth")

# Plot graphs
plotLearningGraphs(learning_stats)
print()
dqn_avg_rew = dqn_agent.test_model(10)
rand_avg_rew = agent.test_model(10)
print("Average reward DQN Agent: ", dqn_avg_rew)
print("Average reward Random Agent: ", rand_avg_rew)

plt.plot(dqn_avg_rew)
plt.ylabel("Total reward DQN agent")
plt.show()
plt.plot(rand_avg_rew)
plt.ylabel("Total reward random agent")
plt.show()

#env.testFeatureExtraction()
#env.runFeatureExtraction()