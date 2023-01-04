import time
import gym
from gym.utils.play import play
from RandomAgent import RandomAgent
import matplotlib.pyplot as plt
import random 


class Env:

    def __init__(self, environment, render_mode='human', obs_type='rgb'):
        self.gym_env = self.makeEnv(environment)
        self.gym_env.render_mode = render_mode
        self.gym_env.env.obs_type = obs_type
        self.number_of_features=4
    
    # Function to test feature extraction of skiing while implementing 
    def testFeatureExtraction(self):
        # Returns episode in data struct (list(trajectory), sum_reward) for testing
        episode = RandomAgent(self).generateEpisode()
        states = random.choices(episode[0], k=10)
        count=1
        for state in states:
            state_obs = state[0]
            # Data structure of objects:
            # dict(object_type, list( dict( pixel_coord, rgb_value)))
            features = self.features(None, state_obs)
            print()
            print("Features for random state",count,":")
            print(features)
            print()
            self.plot2Observation(state_obs)
            count+=1

    def plot2Observation(self,observation):
        plt.axis("on")
        plt.imshow(observation)
        plt.show()

    def reset(self):
        observation, terminal = self.gym_env.reset()
        return self.features(None, observation), terminal

    def calculate_player_speeds(self, p_obs_objects, n_obs_objects):
        # Code to be implemented
        return 0,0

    def calculate_flag_distances(self, n_obs_objects):
        player_pos = n_obs_objects["player"][0]
        # Get position of flags which are below player
        poles = n_obs_objects["pole"]
        while len(poles)>0:
            flags = self.getFlagCenter(poles.pop(0), poles.pop(0))
            flag_h_pos=flags[1]-player_pos[1]
            flag_v_pos=flags[0]-player_pos[0]
            # Test if first flag set is next sub goal (not above player)
            if flag_v_pos>0:
                print("Player pos: ",player_pos)
                print("Flags pos: ", flags)
                return flag_h_pos, flag_v_pos
        print("ERROR - CANNOT DETECT POLE POSITION.")

    def getFlagCenter(self, first, second):
        y = round((first[0]+second[0])/2)
        x = round((first[1]+second[1])/2)
        return y,x


    # Function to get object center
    def getObjectCenter(self, pixels_dict):
        pixels = list(pixels_dict.items())
        r_pixel_cords = []
        c_pixel_cords = []
        for pixel in pixels:
            r_pixel_cords.append(pixel[0][0])
            c_pixel_cords.append(pixel[0][1]) 
        length = len(pixels)
        return round(sum(r_pixel_cords)/length),round(sum(c_pixel_cords)/length)
 
    # Function to change object pixels into center coordinates.  
    # dict(object_type, list( (y,x) ))
    def objectsToObjectCoords(self, objects):
        for ob_type in objects:
            object_list=objects[ob_type]
            objects[ob_type]=[self.getObjectCenter(object) for object in object_list]
        return objects

    # self.getObjectCenter to be used here so we only store object centers instead of whole pixels. 
    def detectObjects(self, p_observation, n_observation):
        if p_observation is None:
            objects = self.identifyObjects(n_observation)
            return None, self.objectsToObjectCoords(objects)
        else:
            p_obs = self.identifyObjects(p_observation)
            n_obs = self.identifyObjects(n_observation)
            return self.objectsToObjectCoords(p_obs), self.objectsToObjectCoords(n_obs)

    # Function returns feature space:
    #   Player horizontal speed
    #   Player vertical speed
    #   Flag horizontal distance
    #   Flag vertical distance
    def features(self, p_observation, n_observation):
        p_obs_objects, n_obs_objects = self.detectObjects(p_observation, n_observation)
        if p_obs_objects is None:
            player_hspeed = 0
            player_vspeed = 0
        else: 
            player_hspeed, player_vspeed = self.calculate_player_speeds(p_obs_objects, n_obs_objects)
        flag_h, flag_v = self.calculate_flag_distances(n_obs_objects)
        return player_hspeed, player_vspeed, flag_h, flag_v
    
    # We are performing feature extraction on every step of expience, need to do this for states used for training only! 
    def step(self, action, p_observation):
        n_observation, reward, terminal, info = self.gym_env.step(action)
        return self.features(p_observation, n_observation), reward, terminal, info

    def makeEnv(self, environment):
        return gym.make(environment)

    # Human play environment
    def playEnvironment(self, mapping):
        if mapping is None:
            play(self.gym_env)
        else:
            play(self.gym_env, keys_to_action=mapping)

    # Runs environment using random policy, for visual
    def runEnvironment(self):
        environment = self.gym_env
        environment.reset()  # Reset the environment
        termination = False
        while not termination:
            environment.render()
            environment.step(environment.action_space.sample())
            time.sleep(0.01)  # sleep between each timestep
        environment.close()

    def getObjectColourCategory(self, pixel):
        if pixel == (214, 92, 92):
            return 'player'
        elif pixel in [(66, 72, 200), (184, 50, 50)]:
            return 'pole'
        elif pixel in [(158, 208, 101), (72, 160, 72), (110, 156, 66), (82, 126, 45)]:
            return 'tree'
        else:
            return None

    # Fetches all object pixels
    def identifyObjectPixels(self, observation):
        player_pixels = {}
        pole_pixels = {}
        tree_pixels = {}
        for row in range(len(observation)):
            for col in range(len(observation[row])):
                pixel = observation[row][col]
                # Convert to tuple for comparison
                pixel = (pixel[0], pixel[1], pixel[2])
                pixel_type = self.getObjectColourCategory(pixel)
                if pixel_type == 'player':
                    player_pixels[row, col] = pixel
                elif pixel_type == 'pole':
                    pole_pixels[row, col] = pixel
                elif pixel_type == 'tree':
                    tree_pixels[row, col] = pixel
        return player_pixels, pole_pixels, tree_pixels

    def findAdjacentPixelsRecursively(self, pixel_dict, row, col, object_pixels, ob_type):
        pixel = pixel_dict.pop((row, col), None)
        pixel_type = self.getObjectColourCategory(pixel)
        if pixel_type != ob_type:
            return
        else:
            object_pixels[row, col] = pixel
            self.findAdjacentPixelsRecursively(pixel_dict, row + 1, col, object_pixels, ob_type)
            self.findAdjacentPixelsRecursively(pixel_dict, row - 1, col, object_pixels, ob_type)
            self.findAdjacentPixelsRecursively(pixel_dict, row, col + 1, object_pixels, ob_type)
            self.findAdjacentPixelsRecursively(pixel_dict, row, col - 1, object_pixels, ob_type)

    def identifyObjects(self, observation):
        object_pixel_dicts = self.identifyObjectPixels(observation)
        objects = {'player': [object_pixel_dicts[0]]}
        object_pixels = {}
        for idx in range(1, len(object_pixel_dicts)):
            pixel_dict = object_pixel_dicts[idx]
            if idx == 1:
                ob_type = 'pole'
            elif idx == 2:
                ob_type = 'tree'
            ob_list = objects.setdefault(ob_type, [])
            while len(pixel_dict) > 0:
                # Getting first key in dictionary
                (row, col) = next(iter(pixel_dict))
                self.findAdjacentPixelsRecursively(pixel_dict, row, col, object_pixels, ob_type)
                ob_list.append(object_pixels)
                object_pixels = {}
        return objects
