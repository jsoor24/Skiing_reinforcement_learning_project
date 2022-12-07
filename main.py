from Env import Env
from RandomAgent import RandomAgent
from DQNAgent import DQNAgent
import matplotlib.pyplot as plt
import numpy

def play():
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
    environment.playEnvironment(mapping)

def plotLearningGraphs(learning_stats):
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
env = Env('CartPole-v1')
# Initialise agent using environment.
agent = RandomAgent(env)
# Create DQN agent. 
dqn_agent = DQNAgent(env=env, learning_rate=1e-3, sync_freq=5, replay_buffer_size=256)
# Train agent. 
learning_stats = dqn_agent.train(7000)
print("Saving trained model")
dqn_agent.save_trained_model("cartpole-dqn.pth")
# Plot graphs
#plotLearningGraphs(learning_stats)
print()
print("Average reward DQN Agent: ", dqn_agent.test_model(10))
print("Average reward Random Agent: ", agent.test_model(10))