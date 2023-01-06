from Env import Env
from RandomAgent import RandomAgent
from DQNAgent import DQNAgent

# Initialise skiing environment.
env = Env('CartPole-v1')

# Initialise agent using environment.
agent = RandomAgent(env)

# Create DQN agent.
dqn_agent = DQNAgent(env=env, learning_rate=1e-3, sync_freq=5, replay_buffer_size=256)

# Train the agent
# learning_stats = dqn_agent.train(10000)
# print("Saving trained model")
# dqn_agent.save_trained_model("cartpole-dqn.pth")

# Load the model
dqn_agent.load_pretrained_model("optimal-policy.pth")

print()
print("Average reward DQN Agent: ", dqn_agent.test_model(1, True))
# print("Average reward Random Agent: ", agent.test_model(10, True))
