import time
import gym
from gym.utils.play import play
from RandomAgent import RandomAgent
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import copy
import traceback
import matplotlib.image
import numpy as np
from math import sqrt

class Env:

    def __init__(self, environment, render_mode='human', obs_type='rgb', frameskip=0):
        self.gym_env = self.makeEnv(environment)
        self.gym_env.render_mode = render_mode
        self.gym_env.env.obs_type = obs_type
        self.frameskip = frameskip
        self.number_of_features = 6

    def observationIterationFETest(self):
        print()
        agent = RandomAgent(self)
        episode = agent.generateEpisode()
        trajectory = episode[0]
        state_nums = random.choices(range(1,len(trajectory)),k=10)
        for idx in state_nums:
            print("-----------------------------------------")
            observation = trajectory[idx][0]
            p_observation = trajectory[idx-5][0]
            # Frame skipping
            p_objects=self.detectObjects(p_observation)
            print("Previous state objects:",p_objects)
            print()
            start_time = time.time()
            features=self.features(p_objects,observation,optimised=False)
            print("Unoptimised FE (Using pixel checks): ","--- %s seconds ---" % (time.time() - start_time))
            print("Unoptimised FE:",features)
            print()
            # Rerun object detection on previous state as .pop() in FE removes poles
            p_objects=self.detectObjects(trajectory[idx-1][0])
            start_time = time.time()
            op_objects=self.features(p_objects,observation,optimised=True)
            print("Optimised FE (Using dict): ","--- %s seconds ---" % (time.time() - start_time))
            print("Optimised FE:",features)
            print()
            self.plot2Images(p_observation,observation)

    def plot2Images(self, im1, im2):
        # Initialise figure with 2 sub-plots and show image of first and last observations.
        fig1, axes = plt.subplots(1, 2)
        axes[0].imshow(im1, aspect='auto')
        axes[0].set_title("Previous observation")
        axes[1].imshow(im2, aspect='auto')
        axes[1].set_title("Current observation")
        plt.show()

    def plot2Observation(self, observation):
        plt.axis("on")
        plt.imshow(observation)
        plt.show()

    # Object coords:
    # {'player': [(72, 12)],
    # 'pole': [(67, 50), (67, 82), (171, 39), (171, 71)],
    # 'tree': [(107, 150), (137, 136), (196, 145)]}

    def fixPolePositions(self, poles_list):
        corrected = []
        while len(poles_list) > 0:
            poleA = poles_list.pop(0)
            poleB = poles_list.pop(0)
            fixed_v = max(poleA[0], poleB[0])
            corrected.append((fixed_v, poleA[1]))
            corrected.append((fixed_v, poleB[1]))
        return corrected

    def calculate_player_velocities(self, p_obs_objects, n_obs_objects):
        start_player_pos = p_obs_objects["player"][0]
        end_player_pos = n_obs_objects["player"][0]
        # print("Player start:",start_player_pos)
        # print("Player end:",end_player_pos)
        h_velocity = end_player_pos[1] - start_player_pos[1]
        # print()
        # print("Previous objects:",p_obs_objects)
        # print("Next objects:",n_obs_objects)
        start_poles_pos = self.fixPolePositions(p_obs_objects["pole"])
        end_poles_pos = self.fixPolePositions(n_obs_objects["pole"])

        # Ignore poles that are above the player
        # When they start to go off the top of the screen, their height value doesn't change
        # so will give inaccurate velocities
        if (len(start_poles_pos) == 4 or len(end_poles_pos) == 4):
            start_poles = [pole for pole in start_poles_pos if pole[0] >= start_player_pos[0]]
            end_poles = [pole for pole in end_poles_pos if pole[0] >= end_player_pos[0]]
        else:
            start_poles = start_poles_pos
            end_poles = end_poles_pos

        n_of_start_poles = len(start_poles)
        n_of_end_poles = len(end_poles)

        # This should never happen
        if n_of_start_poles != 4 and n_of_start_poles != 2:
            print("Start poles:", start_poles)
            print("End:", end_poles)
            print("ERROR: Unexpected number of poles in observation")
            print("Observed ", n_of_start_poles, " in first observation")
            return 0, 0

        if n_of_end_poles != 4 and n_of_end_poles != 2:
            print("Start poles:", start_poles)
            print("End:", end_poles)
            print("ERROR: Unexpected number of poles in observation")
            print("Observed ", n_of_end_poles, " in second observation")
            print(end_poles_pos)
            return 0, 0

        # 4 scenarios can occur with the number of poles:
        # (4, 4) 4 poles in each observation (check dist between any)
        # (2, 2) 2 poles in each observation (check dist between any)
        # (2, 4) 2 new poles in second observation (check dist between first poles in both observations)
        # (4, 2) 2 poles no longer visible in second observation (check dist between 'third pole' and 'first pole')
        if n_of_start_poles == 4 and n_of_end_poles == 2:
            v_velocity = start_poles[2][0] - end_poles[0][0]
        else:
            v_velocity = start_poles[0][0] - end_poles[0][0]

        return h_velocity, v_velocity

    def calculate_flag_distances(self, n_obs_objects):
        player_pos = n_obs_objects["player"][0]
        # Get position of flags which are below player
        poles = n_obs_objects["pole"]
        while len(poles) > 0:
            flags = self.getFlagCentre(poles.pop(0), poles.pop(0))
            flag_h_pos = flags[1] - player_pos[1]
            flag_v_pos = flags[0] - player_pos[0]
            # Test if first flag set is next sub goal (not above player)
            if flag_v_pos > 0:
                return flag_h_pos, flag_v_pos
        return 0, 0
    
    def calculate_closet_tree_distance(self, n_obs_objects):
        player_pos = n_obs_objects["player"][0]
        # Get position of trees which are below player
        trees = n_obs_objects["tree"]
        distances = []
        tree_idxs = []
        idx=0
        for tree in trees:
            tree_v_dis = tree[0] - player_pos[0]
            # Test if first flag set is next sub goal (not above player)
            if tree_v_dis > 0:
                tree_h_dis = tree[1] - player_pos[1]
                distances.append(sqrt( tree_h_dis**2 + tree_v_dis**2 ))
                tree_idxs.append(idx)
            idx+=1
        closest_tree_idx = tree_idxs[distances.index(min(distances))]
        closest_tree = trees[closest_tree_idx]
        h_dist = closest_tree[1] - player_pos[1]
        v_dist = closest_tree[0] - player_pos[0]
        return v_dist, h_dist

    # Get centre point between two flags
    def getFlagCentre(self, first, second):
        y = round((first[0] + second[0]) / 2)
        x = round((first[1] + second[1]) / 2)
        return y, x

    # Function to get object centre from pixels
    def getObjectCentre(self, pixels_dict):
        pixels = list(pixels_dict.items())
        r_pixel_cords = []
        c_pixel_cords = []
        for pixel in pixels:
            r_pixel_cords.append(pixel[0][0])
            c_pixel_cords.append(pixel[0][1])
        length = len(pixels)
        return round(sum(r_pixel_cords) / length), round(sum(c_pixel_cords) / length)

    # Function to change object pixels into center coordinates.
    # dict(object_type, list( (y,x) ))
    def objectsToObjectCoords(self, objects):
        for ob_type in objects:
            # list( dict( pixel_coord, rgb_value))
            object_list = objects[ob_type]
            if ob_type == "player":
                objects[ob_type] = [list(object_list[0].keys())[0]]
            else:
                objects[ob_type] = [self.getObjectCentre(object) for object in object_list]
        return objects

    # Wrapper function to allow us to call with p = None
    def detectObjects(self, n_observation,optimised=True):
        #start_time = time.time()
        n_obs = self.identifyObjects(n_observation,optimised=optimised)
        #print("identifyObjects: ","--- %s seconds ---" % (time.time() - start_time))
        if (len(n_obs["pole"]) == 0):
            print("Error no poles detected: ", n_obs)
        return self.objectsToObjectCoords(n_obs)

    # Function returns feature space:
    #   Player horizontal velocity
    #   Player vertical velocity
    #   Flag horizontal distance
    #   Flag vertical distance
    def features(self, p_obs_objects, n_observation, optimised=True):
        #start_time = time.time()
        n_obs_objects = self.detectObjects(n_observation,optimised=optimised)
        if(len(n_obs_objects["player"])==0):
            n_obs_objects["player"]==p_obs_objects["player"]
        n_obs_objects_to_return = copy.deepcopy(n_obs_objects)
        n_obs_objects_for_dist = copy.deepcopy(n_obs_objects)
        n_obs_objects_for_tree_dist = copy.deepcopy(n_obs_objects)
        #print("detectObjects: ","--- %s seconds ---" % (time.time() - start_time))

        if p_obs_objects is None:
            h_velocity = 0
            v_velocity = 0
        else:
            #start_time = time.time()
            try:
                h_velocity, v_velocity = self.calculate_player_velocities(p_obs_objects, n_obs_objects)
                #print("calculate_player_velocities: ","--- %s seconds ---" % (time.time() - start_time))
            except Exception as e:
                print("Previous object: ", p_obs_objects)
                print("New object: ", n_obs_objects)
                matplotlib.image.imsave('errorcase.png', n_observation)
                print("Testing detectobjects:", self.detectObjects(n_observation))
                self.plot2Observation(n_observation)
                traceback.print_exc()
                exit(0)

        #start_time = time.time()
        flag_h, flag_v = self.calculate_flag_distances(n_obs_objects_for_dist)
        tree_h, tree_v = self.calculate_closet_tree_distance(n_obs_objects_for_tree_dist)
        #print("calculate_flag_distances: ","--- %s seconds ---" % (time.time() - start_time))
        return (h_velocity, v_velocity, flag_h, flag_v, tree_h, tree_v), n_obs_objects_to_return

    # We are performing feature extraction on every step of experience
    # need to do this for states used for training only!
    def step(self, action, p_obs_objs):
        n_observation, reward, terminal, info = self.gym_env.step(action)
        for i in range(self.frameskip):
            self.gym_env.step(0)
        return self.features(p_obs_objs, n_observation), reward, terminal, info

    def reset(self):
        observation = self.gym_env.reset()
        return self.features(None, observation), False

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

    def treesGetObjectColourCategory(self, pixel):
        if pixel == (214, 92, 92):
            return 'player'
        elif pixel == (66, 72, 200) or pixel == (184, 50, 50):
            return 'pole'
        elif pixel in [(158, 208, 101), (72, 160, 72), (110, 156, 66), (82, 126, 45)]:
            return 'tree'
        else:
            return None

    def getObjectColourCategory(self, pixel):
        if pixel == (214, 92, 92):
            return 'player'
        elif pixel == (66, 72, 200) or pixel == (184, 50, 50):
            return 'pole'
        elif pixel in [(158, 208, 101), (72, 160, 72), (110, 156, 66), (82, 126, 45)]:
            return 'tree'
        else:
            return None

    # Fetches all pixels associated with one object/colour
    def optimisedIdentifyObjectPixels(self, observation):
        image = np.array(observation)
        color_dict={}
        for row in range(59, 250):
            for col in range(7, 150):
                pixel = observation[row,col]
                pixel = (pixel[0], pixel[1], pixel[2])
                pixels = color_dict.setdefault(pixel,{})
                pixels[(row,col)]=pixel
        player_pixels=color_dict.get((214, 92, 92),{})
        pole_pixels=color_dict.get((66, 72, 200),{})|color_dict.get((184, 50, 50),{})
        tree_pixels=color_dict.get((158, 208, 101),{})|color_dict.get((72, 160, 72),{})|color_dict.get((110, 156, 66),{})|color_dict.get((82, 126, 45),{})
        return player_pixels, pole_pixels, tree_pixels

    # Fetches all pixels associated with one object/colour
    def identifyObjectPixels(self, observation):
        image = np.array(observation)
        player_pixels = {}
        pole_pixels = {}
        tree_pixels = {}
        for row in range(59, len(observation)):
            for col in range(7, 150):
                pixel = image[row,col]
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
    def findAdjacentPixelsRecursively(self, pixel_dict, row, col, object_pixels, ob_type, player_dict):
        ob_pixel = pixel_dict.pop((row, col), None)
        player_pix = player_dict.pop((row, col), None)
        pixel_type = self.getObjectColourCategory(ob_pixel)
        if pixel_type != ob_type and player_pix is None:
            return
        else:
            if player_pix is None:
                object_pixels[row, col] = ob_pixel
            self.findAdjacentPixelsRecursively(pixel_dict, row + 1, col, object_pixels, ob_type, player_dict)
            self.findAdjacentPixelsRecursively(pixel_dict, row - 1, col, object_pixels, ob_type, player_dict)
            self.findAdjacentPixelsRecursively(pixel_dict, row, col + 1, object_pixels, ob_type, player_dict)
            self.findAdjacentPixelsRecursively(pixel_dict, row, col - 1, object_pixels, ob_type, player_dict)

    # Identifies objects from observation using pixel colour categories and populates dictionary
    def identifyObjects(self, observation,optimised=True):
        #start_time = time.time()
        if optimised:
            object_pixel_dicts = self.optimisedIdentifyObjectPixels(observation)
        else:
            object_pixel_dicts = self.identifyObjectPixels(observation)
        #print("identifyObjectPixels: --- %s seconds ---" % (time.time() - start_time))
        player_dict = object_pixel_dicts[0]
        objects = {'player': [player_dict]}
        object_pixels = {}
        for idx in range(1, len(object_pixel_dicts)):
            # Append player dict to check for object seperation by player
            pixel_dict = object_pixel_dicts[idx]
            if idx == 1:
                ob_type = 'pole'
            elif idx == 2:
                ob_type = 'tree'
            ob_list = objects.setdefault(ob_type, [])
            while len(pixel_dict) > 0:
                # Getting first key in dictionary
                (row, col) = next(iter(pixel_dict))
                self.findAdjacentPixelsRecursively(pixel_dict, row, col, object_pixels, ob_type, player_dict.copy())
                ob_list.append(object_pixels)
                object_pixels = {}
        return objects
