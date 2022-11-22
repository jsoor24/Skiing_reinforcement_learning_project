import time
import gym
from gym.utils.play import play

class skiing_env:
    
    def __init__(self, render_mode='human', obs_type='rgb'):
        self.env=gym.make('Skiing-v4')
        self.env.render_mode=render_mode
        self.env.obs_type=obs_type
    
    # Human play environment
    def playEnvironment(self):
        play(self.env)

    # Runs environment using random policy, for visusal
    def runEnvironment(self):
        env = self.env
        env.reset() # Reset the environment
        termination=False
        while not termination:
            env.render()
            env.step(env.action_space.sample())
            time.sleep(0.01) # sleep between each timestep
        env.close()

class agent():

    def __init__(self, skiing):
        self.env = skiing.env
    
    def policy(self, observation):
        possible_actions = self.env.action_space
        # Return random action
        return possible_actions.sample()

    # Generates an episode. Returns trajectory containing list of tuples(observation,action,reward) 
    # and total reward collected by that episode
    def generateEpisode(self):
        env = self.env
        observation = env.reset()
        episode = []
        terminal = False
        sum_of_reward=0
        while not terminal:
            action = self.policy(observation)
            n_observation, reward, terminal, info = env.step(action)
            sum_of_reward=sum_of_reward+reward
            episode.append((observation,action,reward))
            observation = n_observation
        env.close()
        return episode, sum_of_reward

# Initilise skiing environment
skiing = skiing_env()
# Initilise agent using environment
agent = agent(skiing)
ep, reward = agent.generateEpisode()
first_observation = ep[0][0]
print()
print("RGB Observation dimentions: ", first_observation.shape)
print("RGB Observation type: ", type(first_observation))
print()

