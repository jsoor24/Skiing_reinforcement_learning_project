import time
import gym
from gym.utils.play import play
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class skiing_env():
    
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

def getUniqueColourPixels(observation_action):
    rgb_colors = {}
    pixels = observation_action[0]
    for row in pixels:
        for pixel in row:
            # Add color to dictonary. Cannot use array as key so convert to tuple.
            rgb_colors[pixel[0],pixel[1],pixel[2]]=pixel
    return rgb_colors.values()

def getEpisodeColourSpace(episode):
    unique_rgb_colors=[]
    for observation_action in episode:
        # Get unique rgb pixels from state observation.
        rgb_colors = getUniqueColourPixels(observation_action)
        for color in rgb_colors:
            color_tuple = color[0],color[1],color[2]
            if not color_tuple in unique_rgb_colors:
                unique_rgb_colors.append(color_tuple)
    return unique_rgb_colors

def plotObservationImage(state_action):
    plt.imshow(state_action[0],aspect='auto')
    plt.draw()

def plotRgb3dColorSpace(rgb_colors, title, ax):
    x = []
    y = []
    z = []
    cs = []
    for color in rgb_colors:
        x.append(color[0])
        y.append(color[1])
        z.append(color[2])
        c = (color[0] / 255, color[1] / 255, color[2] / 255)
        cs.append(c)
        ax.text(color[0],color[1],color[2],color,color=c)
    ax.scatter(x,y,z,c=cs)
    ax.set_title(title)
    ax.set_xlabel('R axis')
    ax.set_ylabel('G axis')
    ax.set_zlabel('B axis')
    

def investigateRgbObservations(episode):
    first_state_action = episode[0]
    last_state_action = episode[len(episode)-1]
    
    print()
    print("RGB Observation dimentions: ", first_state_action[0].shape)
    print("RGB Observation type: ", type(first_state_action[0]))
    print()
    episode_rgb_colors = getEpisodeColourSpace(episode)
    print("Numbers of unique colours in episode observations: ",len(episode_rgb_colors))
    print()
    
    fig1, axes = plt.subplots(1,2)
    axes[0].imshow(first_state_action[0],aspect='auto')
    axes[0].set_title("First Observation")
    axes[1].imshow(last_state_action[0],aspect='auto')
    axes[1].set_title("Last Observation")

    fig2 = plt.figure()
    # set up the axes for the first plot
    ax = fig2.add_subplot(1, 2, 1, projection='3d')
    plotRgb3dColorSpace(getUniqueColourPixels(first_state_action),"First Observation Colour Space",ax)
    ax = fig2.add_subplot(1, 2, 2, projection='3d')
    plotRgb3dColorSpace(getUniqueColourPixels(last_state_action),"Last Observation Colour Space",ax)
    
    fig3 = plt.figure()
    ax = plt.axes(projection='3d')
    plotRgb3dColorSpace(episode_rgb_colors,"Episode Observation Colour Space", ax)
    
    plt.show()

# Initilise skiing environment.
skiing = skiing_env()
# Initilise agent using environment.
agent = agent(skiing)
episode, reward = agent.generateEpisode()
investigateRgbObservations(episode)




