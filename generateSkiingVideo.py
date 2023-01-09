from Env import Env
from RandomAgent import RandomAgent
from DQNAgent import DQNAgent

# Initialise skiing environment.
env = Env('Skiing-v4',frameskip=4)

# Initialise agent using environment.
# agent = RandomAgent(env)

# Create DQN agent.
agent = DQNAgent(env=env, learning_rate=1e-3, sync_freq=5, replay_buffer_size=256)

# Train the agent
# learning_stats = dqn_agent.train(10000)
# print("Saving trained model")
# dqn_agent.save_trained_model("cartpole-dqn.pth")

# Load the model
# skiing-dqn-fixed-frameskipping-4-edited-CAP-100eps started learning to reach flags but gets stuck by staying still
agent.load_pretrained_model("models/skiing-dqn-fixed-frameskipping-4-edited-CAP-100eps.pth")

print()
print("Average reward DQN Agent: ", agent.test_model(10, True))
# print("Average reward Random Agent: ", agent.test_model(10, True))
