from Env import Env
from RandomAgent import RandomAgent
import pygame


def play():
    environment.playEnvironment(mapping)


# Initialise skiing environment.
mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
environment = Env('CartPole-v1')
# Initialise agent using environment.
agent = RandomAgent(environment)
# Generate episode using agent.
episode, reward = agent.generateEpisode()
print(episode)
print("\n")
print(reward)
