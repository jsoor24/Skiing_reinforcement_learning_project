import env
import pygame
from DQNAgent import DQNAgent

# Initialise skiing environment.
mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
skiing = env.env('Skiing-v4')
#skiing.playEnvironment(mapping)
# Initialise agent using environment.
agent = DQNAgent(skiing)
# Generate episode using agent.
episode, reward = agent.generateEpisode()
env.investigateRgbObservations(episode)
env.testObjectDetection(episode)