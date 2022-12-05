import time
import gym
import random
import pygame
from gym.utils.play import play
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class env():
    
    def __init__(self, env, render_mode='human', obs_type='rgb'):
        self.env=self.makeEnv(env)
        self.env.render_mode=render_mode
        self.env.obs_type=obs_type
    
    def makeEnv(self, env):
        return gym.make(env)
    
    # Human play environment
    def playEnvironment(self, mapping):
        if(mapping==None):
            play(self.env)
        else: play(self.env,keys_to_action=mapping)

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

    def getObjectColourCategory(self, pixel):
        if pixel == (214,92,92):
            return 'player'
        elif pixel in [(66,72,200),(184,50,50)]:
            return 'pole'
        elif pixel in [(158,208,101),(72,160,72),(110,156,66),(82,126,45)]:
            return 'tree'
        else: return None
    
    # Fetches all object pixels
    def identifyObjectPixels(self,observation):
        player_pixels = {}
        pole_pixels = {}
        tree_pixels = {}
        for row in range(len(observation)):
            for col in range(len(observation[row])):
                pixel = observation[row][col]
                # Convert to tuple for comparison
                pixel = (pixel[0],pixel[1],pixel[2])
                pixel_type = self.getObjectColourCategory(pixel)
                if pixel_type == 'player':
                    player_pixels[row,col]=pixel
                elif pixel_type == 'pole':
                    pole_pixels[row,col]=pixel
                elif pixel_type == 'tree':
                    tree_pixels[row,col]=pixel
        return player_pixels, pole_pixels, tree_pixels


    def findAdjacentPixelsRecursively(self, pixel_dict, row, col, object_pixels, ob_type):
        pixel = pixel_dict.pop((row,col),None)
        pixel_type = self.getObjectColourCategory(pixel)
        if pixel_type != ob_type:
            return
        else:
            object_pixels[row,col]=pixel
            self.findAdjacentPixelsRecursively(pixel_dict, row+1, col, object_pixels, ob_type)
            self.findAdjacentPixelsRecursively(pixel_dict, row-1, col, object_pixels, ob_type)
            self.findAdjacentPixelsRecursively(pixel_dict, row, col+1, object_pixels, ob_type)
            self.findAdjacentPixelsRecursively(pixel_dict, row, col-1, object_pixels, ob_type)
            

    def identifyObjects(self, observation):
        object_pixel_dicts = self.identifyObjectPixels(observation)
        objects={}
        objects['player']=[object_pixel_dicts[0]]
        object_pixels={}
        for idx in range(1,len(object_pixel_dicts)):
            pixel_dict = object_pixel_dicts[idx]
            if idx==1:
                ob_type = 'pole'
            elif idx==2:
                ob_type = 'tree'
            ob_list = objects.setdefault(ob_type,[]) 
            while(len(pixel_dict)>0):
                # Getting first key in dictionary
                (row,col) = next(iter(pixel_dict))
                self.findAdjacentPixelsRecursively(pixel_dict, row, col, object_pixels, ob_type)
                ob_list.append(object_pixels)
                object_pixels={}
        return objects

            

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
    # Get first and last state_action pair.
    first_state_action = episode[0]
    last_state_action = episode[len(episode)-1]
    
    print()
    print("RGB Observation dimentions: ", first_state_action[0].shape)
    print("RGB Observation type: ", type(first_state_action[0]))
    print()
    # Get all unique rgb colours seen in episode observations.
    episode_rgb_colors = getEpisodeColourSpace(episode)
    print("Numbers of unique colours in episode observations: ",len(episode_rgb_colors))
    print()
    
    # Initilise figure with 2 sub plots and show image of first and last observations.
    fig1, axes = plt.subplots(1,2)
    axes[0].imshow(first_state_action[0],aspect='auto')
    axes[0].set_title("First Observation")
    axes[1].imshow(last_state_action[0],aspect='auto')
    axes[1].set_title("Last Observation")

    fig2 = plt.figure()
    # Set up the axes for the first 3d colour space plot.
    ax = fig2.add_subplot(1, 2, 1, projection='3d')
    # Get unique rgb colours seen in first observation and plot 3d colour space.
    plotRgb3dColorSpace(getUniqueColourPixels(first_state_action),"First Observation Colour Space",ax)
    # Get unique rgb colours seen in last observation and plot 3d colour space.
    ax = fig2.add_subplot(1, 2, 2, projection='3d')
    # Set up the axes for the last 3d colour space plot.
    plotRgb3dColorSpace(getUniqueColourPixels(last_state_action),"Last Observation Colour Space",ax)
    
    # Plot 3d colour space of unique rgb colors seen in the whole episode.  
    fig3 = plt.figure()
    ax = plt.axes(projection='3d')
    plotRgb3dColorSpace(episode_rgb_colors,"Episode Observation Colour Space", ax)
    
    # Show all figures made. 
    plt.show()

def printNumberOfObjectsDetected(objects):
    print()
    for obs in ['player','pole','tree']:
        print(obs," objects detected: ",len(objects.get(obs,None)))
    print()

def plot2Images(observation,ob_detect):
    # Initilise figure with 2 sub plots and show image of first and last observations.
    fig1, axes = plt.subplots(1,2)
    axes[0].imshow(observation,aspect='auto')
    axes[0].set_title("Observation Image")
    axes[1].imshow(ob_detect,aspect='auto')
    axes[1].set_title("Object Detection")

def pixelDictTo2dRgbArray(pixel_dict):
    object2dImage = [[[255,255,255] for j in range(160)] for i in range(250)]
    for object_type in pixel_dict:
        obs = pixel_dict.get(object_type)
        for ob_pixels in obs:
            for (row,col) in ob_pixels:
                pixel = ob_pixels.get((row,col))
                object2dImage[row][col]=[pixel[0],pixel[1],pixel[2]]
    return object2dImage

def testObjectDetection(episode):
    for i in range(3):
        ep = episode[random.randint(0, len(episode)-1)]
        objects = agent.identifyObjects(ep[0])
        printNumberOfObjectsDetected(objects)
        plot2Images(ep[0], pixelDictTo2dRgbArray(objects))
        plt.show()

    final = episode.pop()
    objects = agent.identifyObjects(final[0])
    printNumberOfObjectsDetected(objects)
    plot2Images(final[0], pixelDictTo2dRgbArray(objects))
    plt.show()


# Initilise skiing environment.
mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
skiing = env('CartPole-v1')
skiing.playEnvironment(mapping)
# Initilise agent using environment.
#agent = agent(skiing)
# Generate episode using agent.
#episode, reward = agent.generateEpisode()
#investigateRgbObservations(episode)
#testObjectDetection(episode)