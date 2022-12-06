from env import env
from DQNAgent import DQNAgent
import pygame

def play():
    skiing.playEnvironment(mapping)

# Initialise skiing environment.
mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
environement = env('Skiing-v4')
# Initialise agent using environment.
agent = DQNAgent(environement)
# Generate episode using agent.
episode, reward = agent.generateEpisode()
