import time
import gym
from gym.utils.play import play


class Env:

    def __init__(self, environment, render_mode='human', obs_type='rgb'):
        self.env = self.makeEnv(environment)
        self.env.render_mode = render_mode
        self.env.obs_type = obs_type

    def makeEnv(self, environment):
        return gym.make(environment)

    # Human play environment
    def playEnvironment(self, mapping):
        if mapping is None:
            play(self.env)
        else:
            play(self.env, keys_to_action=mapping)

    # Runs environment using random policy, for visual
    def runEnvironment(self):
        environment = self.env
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
