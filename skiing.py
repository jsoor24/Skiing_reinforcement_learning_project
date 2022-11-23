import time
import gym
from gym.utils.play import play
import matplotlib.pyplot as plt

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
    def generateEpisode(self, render=False):
        env = self.env
        observation = env.reset()
        episode = []
        terminal = False
        sum_of_reward=0
        while not terminal:
            # If been told to render, render before every action so user can see simulation. 
            if(render):
                env.render()
            action = self.policy(observation)
            n_observation, reward, terminal, info = env.step(action)
            sum_of_reward=sum_of_reward+reward
            episode.append((observation,action,reward))
            observation = n_observation
        env.close()
        return episode, sum_of_reward

def getUniqueColourPixels(episode):
    rgb_colors = {}
    # Loop through state observations in episode. 
    for observation in range(len(episode)):
        # Get rgb pixels from state observation.
        pixels = episode[observation][0]
        for row in pixels:
            for pixel in row:
                # Add color to dictonary. Cannot use array as key so convert to tuple.
                rgb_colors[pixel[0],pixel[1],pixel[2]]=pixel
    return rgb_colors.values()

def plot3dColorSpace(rgb_colors):
    ax = plt.axes(projection='3d')
    x = []
    y = []
    z = []
    cs = []
    for color in rgb_colors:
        print(color)
        x.append(color[0])
        y.append(color[1])
        z.append(color[2])
        c = (color[0] / 255, color[1] / 255, color[2] / 255)
        cs.append(c)
        ax.text(color[0],color[1],color[2],color,color=c)
    ax.scatter(x,y,z,c=cs)
    plt.title("Skiing 3D Color Space")
    ax.set_xlabel('R axis')
    ax.set_ylabel('G axis')
    ax.set_zlabel('B axis')
    plt.show()

# Initilise skiing environment.
skiing = skiing_env()
# Initilise agent using environment.
agent = agent(skiing)
episode, reward = agent.generateEpisode()
first_observation = episode[0][0]
print()
print("RGB Observation dimentions: ", first_observation.shape)
print("RGB Observation type: ", type(first_observation))
print()
rgb_colors = getUniqueColourPixels(episode)
print("Numbers of unique colours in episode observations: ",len(rgb_colors))
plot3dColorSpace(rgb_colors)
print()




