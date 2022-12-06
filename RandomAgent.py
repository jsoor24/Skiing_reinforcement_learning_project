class RandomAgent:

    def __init__(self, skiing):
        self.env = skiing.env

    def policy(self, observation):
        possible_actions = self.env.action_space
        # Return random action
        return possible_actions.sample()

    # Generates an episode. Returns trajectory containing list of tuples(observation,action,reward) 
    # and total reward collected by that episode
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
