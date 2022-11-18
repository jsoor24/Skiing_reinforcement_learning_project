import gym
import time
from gym.utils.play import play

def createEnvironment():
    env = gym.make('Skiing-v4')
    return env

# Runs environment using random policy
def runEnvironment(env):
    observation = env.reset()# Reset the environment
    termination=False
    while not termination:
        env.render('human') # Render the environment
        observation, reward, termination, info = doRandomAction(env)
        time.sleep(0.01) # sleep between each timestep
    env.close()

def doRandomAction(env):
    return env.step(env.action_space.sample())

# Human play environment
def playEnvironment(env):
    play(env)

env = createEnvironment()
runEnvironment(env)
#playEnvironment(env)
