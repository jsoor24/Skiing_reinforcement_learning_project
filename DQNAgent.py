class DQNAgent:

    def __init__(self):
        # some code
        return

    def policy(self, observation):
        # some code
        return

    def generateEpisode(self, render=False):
        env = self.env
        observation = env.reset()
        episode = []
        terminal = False
        sum_of_reward = 0
        while not terminal:
            # If been told to render, render before every action so user can see simulation.
            if render:
                env.render()
            action = self.policy(observation)
            n_observation, reward, terminal, info = env.step(action)
            sum_of_reward = sum_of_reward + reward
            episode.append((observation, action, reward))
            observation = n_observation
        env.close()
        return episode, sum_of_reward
