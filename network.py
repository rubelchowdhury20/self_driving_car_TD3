import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
import gym
import torch
import torch.nn as nn

import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque


# Step 1: We initialize the Experience Replay memory

class ReplayBuffer(object):

    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

    def length(self):
        return len(self.storage)
# Step 2: We build one neural network for the Actor model and one neural network for the Actor target

# Actor input is patch image (say 40x 40)
# Output is action (which is 3 orientation value for us)


class Actor(nn.Module):

    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 16, 1)

        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 16, 1)

        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(8)

        self.layer_1 = nn.Linear(16, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x
# Step 3: We build two neural networks for the two Critic models and two neural networks for the two Critic targets


class Critic(nn.Module):
    def __init__(self, action_dim):
        super(Critic, self).__init__()

        def conv_block_first(in_dim, out_dim):
            model = nn.Sequential(
                nn.Conv2d(in_dim, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_dim), nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(.2),
                nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_dim), nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(.2),
                nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.AvgPool2d(8),
                nn.Linear(16, out_dim))
            return model

        # convert image to a small dim
        # first dim --> number of input channel
        # 2nd dim --> output neurons
        self.conv_img = conv_block_first(3, 32)
        state_dim = 32
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = self.conv_img(x)
        print(x.shape, u.shape)
        xu = torch.cat([x, u], 1)
        # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Forward-Propagation on the second Critic Neural Network
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, u):
        x = self.conv_img(x)
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

# Steps 4 to 15: Training Process
# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class


class TD3(object):


    

    def __init__(self, action_dim, max_action):
        self.actor = Actor(action_dim, max_action).to(device)
        self.actor_target = Actor(action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(action_dim).to(device)
        self.critic_target = Critic(action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action

    def select_action(self, state):
        probs = F.softmax(self.actor_target(next_state) * 100)
        action = probs.multinomial()

        return action.data[0, 0]

        

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):


            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(
                batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Step 5: From the next state s’, the Actor target plays the next action a’
            next_action = self.actor_target(next_state)

            # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(
                0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (
                next_action + noise).clamp(-self.max_action, self.max_action)

            # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)

            # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            current_Q1, current_Q2 = self.critic(state, action)

            # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(
                current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % policy_freq == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data)

                # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data)

    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' %
                   (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' %
                   (directory, filename))

    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(
            '%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load(
            '%s/%s_critic.pth' % (directory, filename)))

    
env_name = "AntBulletEnv-v0" # Name of a environment (set it to any Continous environment you want)
seed = 0 # Random seed number
start_timesteps = 1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # 

class train_t3d():

    def __init__(self):
        ## We set the parameters       
        self.reward_window = []


        self.last_state = torch.Tensor((3,40,40)).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0


        ## We create a file name for the two saved models: the Actor and Critic models
        self.file_name = "%s_%s_%s" % ("TD3", 'td3', str(seed))
        print ("---------------------------------------")
        print ("Settings: %s" % (file_name))
        print ("---------------------------------------")

        ## We create a folder inside which will be saved the trained models
        if not os.path.exists("./results"):
            os.makedirs("./results")
        if save_models and not os.path.exists("./pytorch_models"):
            os.makedirs("./pytorch_models")

        # env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        # state_dim = env.observation_space.shape[0]
        # action_dim = env.action_space.shape[0]
        # max_action = float(env.action_space.high[0])


        self.action_dim = 3
        self.max_action = 1 ## todo

        self.policy = TD3(self.action_dim, self.max_action)

        self.replay_buffer = ReplayBuffer()

        self.total_timesteps = 0
        self.timesteps_since_eval = 0
        self.episode_num = 0
        self.done = True
        self.t0 = time.time()
    


    def update(self, reward, new_image):
        new_image = torch.Tensor(new_image).float().unsqueeze(0)
        self.replay_buffer.add((self.last_state,new_image,self.last_action, self.last_reward,_))
        action = self.policy.select_action(new_image)
        if len(self.replay_buffer) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.replay_buffer(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_image
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action





