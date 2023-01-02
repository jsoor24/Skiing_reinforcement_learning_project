import torch
import copy
import numpy
from collections import deque
import random
from tqdm import tqdm

class DQNAgent:

    def __init__(self, env, learning_rate, sync_freq, replay_buffer_size):
        # Manual seed used when generating initial weights for nn. Used to ensure converging, not sure why! 
        torch.manual_seed(1234)
        self.env=env.gym_env
        nn_layer_sizes = self.getLayerSizes()

        # create two NN networks, one for action-values, one for target action-values.
        self.q_action_values_nn = self.build_nn(nn_layer_sizes)
        self.q_target_values_nn = copy.deepcopy(self.q_action_values_nn)

        # Get MSE loss function.
        self.loss_fn = torch.nn.MSELoss()
        # Get NN optimiser function.
        self.optimiser = torch.optim.Adam(self.q_action_values_nn.parameters(),lr=learning_rate)

        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.gamma = torch.tensor(0.95).float()

        # Initilise experience replay 
        self.replay_buffer = self.initiliseReplayBuffer(replay_buffer_size)
        print()
        print("DQN Agent created ")
        return
    
    def load_pretrained_model(self, model_path):
        self.q_action_values_nn.load_state_dict(torch.load(model_path))

    def save_trained_model(self, model_path="cartpole-dqn.pth"):
        torch.save(self.q_action_values_nn.state_dict(), model_path)

    def sample_from_replay_buffer(self, sample_size):
        # If requested sample size is less than replay_buffer size,
        # then set sample size equal to buffer size. This should not happen.
        if(len(self.replay_buffer) < sample_size):
            sample_size = len(self.experience_replay) 
        # Get sample from replay buffer
        sample = random.sample(self.replay_buffer, int(sample_size))
        # Take each state, action, reward and n_state of each time step 
        # and put into individual lists
        s = torch.tensor(numpy.array([exp[0] for exp in sample])).float()
        a = torch.tensor(numpy.array([exp[1] for exp in sample])).float()
        r = torch.tensor(numpy.array([exp[2] for exp in sample])).float()
        n_s = torch.tensor(numpy.array([exp[3] for exp in sample])).float()   
        return s, a, r, n_s

    def get_q_next(self, states):
        with torch.no_grad():
            qp = self.q_target_values_nn(states)
        q,_ = torch.max(qp, axis=1)    
        return q

    def trainNNs(self, batch_size):
        states, actions, rewards, n_states = self.sample_from_replay_buffer(batch_size)

        # Update target network if sync counter == sync frequency
        if (self.network_sync_counter == self.network_sync_freq):
            self.network_sync_counter=0
            # Load q value network parameters (weights) into q target value network
            self.q_target_values_nn.load_state_dict(self.q_action_values_nn.state_dict())

        # Predict return of states using main q_value network
        q_values = self.q_action_values_nn(states)
        pred_max_q_values, _ = torch.max(q_values,axis=1)

        # Get next returns using target network
        next_q_values = self.get_q_next(n_states)
        # Additional reward function
        #rewards = reward_function(states, actions, rewards)
        target_returns = rewards + self.gamma * next_q_values

        # Update weights of main q_value network
        loss = self.loss_fn(pred_max_q_values, target_returns)
        self.optimiser.zero_grad()
        loss.backward(retain_graph=True)
        self.optimiser.step()
        
        self.network_sync_counter += 1       
        return loss.item()

    def train(self, training_episodes):
        print("")
        print("DQN Agent: starting training")
        print("")
        epsilon = 1
        # Variable to test if replay buffer has been half filled. Set to half of capcity to begin with, 
        # so after first time-step, training begins.
        buffer_idx = self.replay_buffer.maxlen/2
        # Lists for analysis of performance. 
        losses_list, reward_list, episode_len_list, epsilon_list  = [], [], [], []
        # Enter loop with progress bar. 
        print("Progress:")
        for ep in tqdm(range(training_episodes)):
            observation, terminal, sum_rewards, ep_len, losses = self.env.reset(), False, 0, 0, 0
            while not terminal:
                ep_len+=1
                action = self.policy(observation, epsilon)
                n_observation, reward, terminal, info = self.env.step(action)
                # Collect experience by adding to replay buffer.
                self.replay_buffer.append((observation, action, reward, n_observation))

                sum_rewards += reward
                observation = n_observation
                buffer_idx += 1

                # If capacity of replay buffer is half filled.
                if(buffer_idx>(self.replay_buffer.maxlen/2)):
                    buffer_idx = 0
                    for i in range(4):
                        loss = self.trainNNs(batch_size=self.replay_buffer.maxlen/4)
                        losses+=loss
            # As we explore, reduce exploration to exploitation.
            if epsilon > 0.05 :
                epsilon -= (1 / (training_episodes/2))
            losses_list.append(losses/ep_len), reward_list.append(sum_rewards), episode_len_list.append(ep_len), epsilon_list.append(epsilon)
        self.env.close()
        print()
        print("TRAINING COMPLETED")
        return losses_list, reward_list, episode_len_list, epsilon_list
        

    # Return replay buffer with random experience. 
    def initiliseReplayBuffer(self, replay_buffer_size):
        replay_buffer = deque(maxlen = replay_buffer_size)
        while(len(replay_buffer)<replay_buffer.maxlen):
            observation = self.env.reset()
            terminal = False
            while not terminal:
                action = self.policy(observation, epsilon=1)
                n_observation, reward, terminal, info = self.env.step(action)
                # Collect experience by adding to replay buffer
                replay_buffer.append((observation, action, reward, n_observation))
                observation = n_observation
        return replay_buffer
    
    def getLayerSizes(self):
        ob_space = self.env.observation_space.shape[0]
        action_space = self.env.action_space.n
        return ob_space, 64, action_space
    
    def build_nn(self, nn_layer_sizes):
        # Create list of nn layers and activation functions.
        layers = []
        # Loop through nn dimentions.
        for idx in range(len(nn_layer_sizes)-1):
            # Create nn layer of input and output size.
            layer = torch.nn.Linear(nn_layer_sizes[idx],nn_layer_sizes[idx+1])
            # If layer is not the last, then use tanH as activation function. 
            if(idx<len(nn_layer_sizes)-2):
                act_function = torch.nn.Tanh()
            else:
                act_function = torch.nn.Identity()
            # Append layer then act_function to layer list.
            layers+=(layer,act_function)
        # Return sequential model of nn layers.
        return torch.nn.Sequential(*layers)

    # Joe's original policy
    # def policy(self, obs, epsilon):
    #     actions = self.env.action_space
    #     if(random.choices((True,False),(epsilon,1-epsilon))[0]):
    #         return actions.sample()
    #     with torch.no_grad():
    #         q_values = self.q_action_values_nn(torch.from_numpy(obs).float())
    #     Q,A = torch.max(q_values, dim=0)
    #     return A.item()

    # From blog policy
    # def policy(self, obs, epsilon):
    #     # We do not require gradient at this point, because this function will be used either
    #     # during experience collection or during inference
    #     with torch.no_grad():
    #         Qp = self.q_action_values_nn(torch.from_numpy(obs).float())
    #     Q,A = torch.max(Qp, axis=0)
    #     A = A if torch.rand(1,).item() > epsilon else torch.randint(0,self.env.action_space.n,(1,))
    #     return A.item()

    # Jeev's rearranged blog policy
    def policy(self, obs, epsilon):
        if torch.rand(1,).item() > epsilon:
            with torch.no_grad():
                Qp = self.q_action_values_nn(torch.from_numpy(obs).float())
            Q, A = torch.max(Qp, axis=0)
            return A.item()
        return torch.randint(0, self.env.action_space.n, (1,)).item()

    def test_model(self, ep_num, render=False):
        episodes = []
        for i in tqdm(range(ep_num)):
            episodes.append(self.generateEpisode(0, render)[1])
        return episodes
        # return sum(episodes)/len(episodes)

    def generateEpisode(self, epsilon, render=False):
        observation = self.env.reset()
        episode = []
        terminal = False
        sum_of_reward = 0
        while not terminal:
            # If been told to render, render before every action so user can see simulation.
            if render:
                self.env.render()
            action = self.policy(observation, epsilon)
            n_observation, reward, terminal, info = self.env.step(action)
            sum_of_reward = sum_of_reward + reward
            episode.append((observation, action, reward, n_observation))
            observation = n_observation
        self.env.close()
        return episode, sum_of_reward
