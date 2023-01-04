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

        # For random time steps
        # states = random.choices(episode[0], k=10)
        # For x time steps at a time
        states = []
        for i in range(0, 100, 10):
            states.append(episode[0][i])


        count=1
        p_state_obs = None
        prev_flag = []
        prev_pos = []
        for state in states:
            n_state_obs = state[0]
            # Data structure of objects:
            # dict(object_type, list( dict( pixel_coord, rgb_value)))
            features = self.features(p_state_obs, n_state_obs)

            if p_state_obs is not None:
                p_objects, n_objects = self.detectObjects(p_state_obs, n_state_obs)
                prev_flag = p_objects["pole"]
                prev_pos = p_objects["player"][0]
            else:
                p_objects, n_objects = self.detectObjects(None, n_state_obs)

            next_flag = n_objects["pole"]
            next_pos = n_objects["player"][0]

            print("Flags: ", prev_flag, " -> ", next_flag)
            print("Player: ", prev_pos, " -> ", next_pos)
            print("--------")
            print("Features for random state",count,":")
            print(features)
            print()

            self.plot2Observation(n_state_obs)
            p_state_obs = n_state_obs
            count+=1

    def plot2Observation(self,observation):
        plt.axis("on")
        plt.imshow(observation)
        plt.show()

    def reset(self):
        observation, terminal = self.gym_env.reset()
        return self.features(None, observation), terminal

    # Object coords:
    # {'player': [(72, 12)],
    # 'pole': [(67, 50), (67, 82), (171, 39), (171, 71)],
    # 'tree': [(107, 150), (137, 136), (196, 145)]}

    def calculate_player_velocities(self, p_obs_objects, n_obs_objects):
        # Code to be implemented
        start_player_pos = p_obs_objects["player"][0]
        end_player_pos = n_obs_objects["player"][0]
        h_velocity = end_player_pos[1] - start_player_pos[1]

        start_poles_pos = p_obs_objects["pole"]
        end_poles_pos = n_obs_objects["pole"]

        # Take care of scenario where 4 flags are disappearing from top of screen
        # ignore flags above player
        # play the game and double check

        n_of_start_poles = len(start_poles_pos)
        n_of_end_poles = len(end_poles_pos)

        # This should never happen
        if n_of_start_poles != 4 and n_of_start_poles != 2:
            print("ERROR: Unexpected number of poles in observation")
            print("Observed ", n_of_start_poles, " in first observation")
            return 0, 0

        if n_of_end_poles != 4 and n_of_end_poles != 2:
            print("ERROR: Unexpected number of poles in observation")
            print("Observed ", n_of_end_poles, " in second observation")
            return 0, 0

        # 4 scenarios can occur with the number of poles:
        # (4, 4) 4 poles in each observation (check dist between any)
        # (2, 2) 2 poles in each observation (check dist between any)
        # (2, 4) 2 new poles in second observation (check dist between first poles in both observations)
        # (4, 2) 2 poles no longer visible in second observation (check dist between 'third pole' and 'first pole')
        if n_of_start_poles == 4 and n_of_end_poles == 2:
            v_velocity = end_poles_pos[0][0] - start_poles_pos[2][0]
        else:
            v_velocity = end_poles_pos[0][0] - start_poles_pos[0][0]

        return h_velocity, v_velocity

    def calculate_flag_distances(self, n_obs_objects):
        player_pos = n_obs_objects["player"][0]
        # Get position of flags which are below player
        poles = n_obs_objects["pole"]
        while len(poles)>0:
            flags = self.getFlagCentre(poles.pop(0), poles.pop(0))
            flag_h_pos=flags[1]-player_pos[1]
            flag_v_pos=flags[0]-player_pos[0]
            # Test if first flag set is next sub goal (not above player)
            if flag_v_pos>0:
                return flag_h_pos, flag_v_pos
        print("ERROR - CANNOT DETECT POLE POSITION.")

    # Get centre point between two flags
    def getFlagCentre(self, first, second):
        y = round((first[0]+second[0])/2)
        x = round((first[1]+second[1])/2)
        return y,x


    # Function to get object center from pixels
    def getObjectCentre(self, pixels_dict):
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
            objects[ob_type]=[self.getObjectCentre(object) for object in object_list]
        return objects

    # Wrapper function to allow us to call with p = None
    def detectObjects(self, p_observation, n_observation):
        if p_observation is None:
            objects = self.identifyObjects(n_observation)
            return None, self.objectsToObjectCoords(objects)
        else:
            p_obs = self.identifyObjects(p_observation)
            n_obs = self.identifyObjects(n_observation)
            return self.objectsToObjectCoords(p_obs), self.objectsToObjectCoords(n_obs)

    # Function returns feature space:
    #   Player horizontal velocity
    #   Player vertical velocity
    #   Flag horizontal distance
    #   Flag vertical distance
    def features(self, p_observation, n_observation):
        p_obs_objects, n_obs_objects = self.detectObjects(p_observation, n_observation)
        if p_obs_objects is None:
            h_velocity = 0
            v_velocity = 0
        else:
            h_velocity, v_velocity = self.calculate_player_velocities(p_obs_objects, n_obs_objects)
        flag_h, flag_v = self.calculate_flag_distances(n_obs_objects)
        return h_velocity, v_velocity, flag_h, flag_v

    # We are performing feature extraction on every step of experience, need to do this for states used for training only!
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

    # Fetches all pixels associated with one object/colour
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

    # Finds neighbouring pixels for one object
    # Given one blue dot for player identifies all player object pixels
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

    # Identifies objects from observation using pixel colour categories and populates dictionary
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
