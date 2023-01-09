from Env import Env
from RandomAgent import RandomAgent
from DQNAgent import DQNAgent
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def play():
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
    environment.playEnvironment(mapping)


def plotLearningGraphs(learning_stats):
    print("Plotting graphs")
    for idx in range(len(learning_stats)):
        plt.plot(learning_stats[idx])
        if (idx == 0):
            ylabel = "Loss difference"
        elif (idx == 1):
            ylabel = "Total reward"
        elif (idx == 2):
            ylabel = "Episode length"
        elif (idx == 3):
            ylabel = "Epsilon"
        plt.ylabel(ylabel)
        plt.show()
    return

# read an image
def test_error_case(env):
    observation = mpimg.imread('errorcase.png')
    print(observation)
    objects = env.identifyObjects(observation)
    print(objects)

def plotModelGraphs():
    # skiing-dqn-fixed-frameskipping-4-edited-CAP-100eps started learning to reach flags but gets stuck by staying still.
    # skiing-dqn-fixed-frameskipping-4-edited-CAP-ZEROVELOCITY-100eps.pth learned to go straight down
    # skiing-dqn-fixed-frameskipping-4-edited-CAP-ZEROVELOCITY-NOTSTRAIGHTDOWN-100eps.pth gets stick immediately 
    # skiing-dqn-frameskipping-5.pth learned to go all the way right 8000eps
    # skiing-dqn-frameskipping-5-edited-CAP.pth learned to go all the way left
    env1 = Env('Skiing-v4',frameskip=4)
    env2 = Env('Skiing-v4',frameskip=5)
    dqn_best = DQNAgent(env=env1, learning_rate=1e-3, sync_freq=5, replay_buffer_size=256)
    dqn_best.load_pretrained_model("models/skiing-dqn-fixed-frameskipping-4-edited-CAP-100eps.pth")
    dqn_left = DQNAgent(env=env2, learning_rate=1e-3, sync_freq=5, replay_buffer_size=256)
    dqn_left.load_pretrained_model("models/skiing-dqn-frameskipping-5-edited-CAP.pth")
    dqn_straight_down = DQNAgent(env=env1, learning_rate=1e-3, sync_freq=5, replay_buffer_size=256)
    dqn_straight_down.load_pretrained_model("models/skiing-dqn-fixed-frameskipping-4-edited-CAP-ZEROVELOCITY-100eps.pth")
    random_agent = RandomAgent(env1)
    agent_results = {}
    agent_results["dqn_best"]=dqn_best.test_model(10)
    agent_results["dqn_left"]=dqn_left.test_model(10)
    agent_results["dqn_straight_down"]=dqn_straight_down.test_model(10)
    agent_results["random_agent"]=dqn_straight_down.test_model(10)
    dqn_avg_rew = dqn_agent.test_model(10)
    rand_avg_rew = agent.test_model(20)



# Initialise skiing environment.
# env = Env('Skiing-v4',frameskip=4)

# # Initialise agent using environment.
# agent = RandomAgent(env)
# # # Create DQN agent.
# dqn_agent = DQNAgent(env=env, learning_rate=1e-2, sync_freq=5, replay_buffer_size=256)

# # # Train agent.
# # learning_stats = dqn_agent.train(100)
# # print(learning_stats)
# # print("Saving trained model")
# # dqn_agent.save_trained_model("models/skiing-dqn-0901-5.pth")

# # Load the agent mode
# dqn_agent.load_pretrained_model("models/skiing-dqn-fixed-frameskipping-4-edited-CAP-100eps.pth")

# # Plot graphs
# #plotLearningGraphs(learning_stats)
# #print()
# dqn_avg_rew = dqn_agent.test_model(20)
# rand_avg_rew = agent.test_model(20)
# print("Average reward DQN Agent: ", dqn_avg_rew)
# print("Average reward Random Agent: ", rand_avg_rew)

# plt.plot(dqn_avg_rew)
# plt.ylabel("Total reward DQN agent")
# plt.show()
# plt.plot(rand_avg_rew)
# plt.ylabel("Total reward random agent")
# plt.show()
#agent.generateEpisode(render=True, frameskip=True)
#env.observationIterationFETest()
plotModelGraphs()
