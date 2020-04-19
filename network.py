# standard library import

# third party imports
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
	def __init__(self, max_action):
		super(Actor, self).__init__()
		self.main = nn.Sequential(
			# input is 1 x 80 x 80
			nn.Conv2d(1, 8, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(8),

			nn.Conv2d(8, 16, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(16),

			nn.Conv2d(16, 32, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(32),

			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout(p=0.8, inplace=True),

			nn.Conv2d(32, 8, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(8),

			nn.Conv2d(8, 16, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(16),

			nn.Conv2d(16, 24, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(24),

			nn.Conv2d(24, 32, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(32),

			nn.Conv2d(32, 48, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(48),

			nn.Conv2d(48, 16, 1, 1, 0, bias=False),

			nn.AvgPool2d(kernel_size = 7),
			nn.Flatten())

		self.actor_linear_1 = nn.Linear(17, 1)
			
		self.max_action = max_action

	def forward(self, x, y):
		x = self.main(x)
		x = torch.cat([x, y], 1)
		x = self.actor_linear_1(x)

		x = self.max_action * torch.tanh(x)
		# print(x)
		return x

class Critic(nn.Module):
	def __init__(self):
		super(Critic, self).__init__()

		# defining the first critic neural network
		self.critic_1_cnn = nn.Sequential(
			# input is 1 x 80 x 80
			nn.Conv2d(1, 8, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(8),

			nn.Conv2d(8, 16, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(16),

			nn.Conv2d(16, 32, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(32),

			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout(p=0.8, inplace=True),

			nn.Conv2d(32, 8, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(8),

			nn.Conv2d(8, 16, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(16),

			nn.Conv2d(16, 24, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(24),

			nn.Conv2d(24, 32, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(32),

			nn.Conv2d(32, 48, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(48),

			nn.AvgPool2d(kernel_size = 7),
			nn.Flatten())
		self.critic_1_linear_1 = nn.Linear(49, 24)
		self.critic_1_linear_2 = nn.Linear(24, 1)

		# defining the first critic neural network
		self.critic_2_cnn = nn.Sequential(
			# input is 1 x 80 x 80
			nn.Conv2d(1, 8, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(8),

			nn.Conv2d(8, 16, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(16),

			nn.Conv2d(16, 32, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(32),

			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout(p=0.8, inplace=True),

			nn.Conv2d(32, 8, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(8),

			nn.Conv2d(8, 16, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(16),

			nn.Conv2d(16, 24, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(24),

			nn.Conv2d(24, 32, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(32),

			nn.Conv2d(32, 48, 3, 1, 0, bias=False),
			nn.ReLU(True),
			nn.BatchNorm2d(48),

			nn.AvgPool2d(kernel_size = 7),
			nn.Flatten())
		self.critic_2_linear_1 = nn.Linear(49, 24)
		self.critic_2_linear_2 = nn.Linear(24, 1)

	def forward(self, x, u):
		# forward propagation through the first critic model
		x1 = self.critic_1_cnn(x)
		x1 = torch.cat([x1, u], 1)
		x1 = self.critic_1_linear_1(x1)
		x1 = self.critic_1_linear_2(x1)

		# forward propagation through the second critic model
		x2 = self.critic_2_cnn(x)
		x2 = torch.cat([x2, u], 1)
		x2 = self.critic_2_linear_1(x2)
		x2 = self.critic_2_linear_2(x2)

		return x1, x2

	def q1(self, x, u):
		x = self.critic_1_cnn(x)
		x = torch.cat([x, u], 1)
		x = self.critic_1_linear_1(x)
		x = self.critic_1_linear_2(x)

		return x

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

	def length(self):
		return len(self.storage)
			
	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		batch_states, batch_orientations, batch_next_states, batch_next_orientations, batch_actions, batch_rewards, batch_dones = [], [], [], [], [], [], []
		for i in ind:
			state, orientation, next_state, next_orientation, action, reward, done = self.storage[i]
			batch_states.append(np.array(state, copy=False))
			batch_orientations.append(np.array(orientation, copy=False))
			batch_next_states.append(np.array(next_state, copy=False))
			batch_next_orientations.append(np.array(orientation, copy=False))
			batch_actions.append(np.array(action, copy=False))
			batch_rewards.append(np.array(reward, copy=False))
			batch_dones.append(np.array(done, copy=False))
			
		return np.array(batch_states), np.array(batch_orientations).reshape(-1, 1), np.array(batch_next_states), np.array(batch_next_orientations).reshape(-1, 1), np.array(batch_actions).reshape(-1, 1), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


# Implementing TD3
class TD3(object):
	def __init__(self, max_action):
		self.actor = Actor(max_action).to(device)
		self.actor_target = Actor(max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
		self.critic = Critic().to(device)
		self.critic_target = Critic().to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
		self.replay_buffer = ReplayBuffer()
		self.max_action = max_action

		self.last_state = torch.ones((1, 40, 40))
		self.last_orientation = 0
		self.last_action = 0
		self.last_reward = 0
		self.last_done = False
		self.episode_reward = 0
		self.episode_timesteps = 0

	def select_action(self, state, orientation):
		state_img = torch.tensor(state).float().unsqueeze(0).to(device)
		state_o = torch.tensor(orientation).float().unsqueeze(0).unsqueeze(0).to(device)

		return int(self.actor(state_img, state_o).cpu().data.numpy().flatten().clip(-self.max_action, self.max_action))

	def update(self, reward, new_state, orientation, done):
		self.replay_buffer.add((self.last_state, self.last_orientation, new_state, orientation, self.last_action, self.last_reward, self.last_done))
		if self.replay_buffer.length() < 3000:
			action = int(np.random.randint(-5, 5, 1)[0])
		else:
			action = self.select_action(new_state, orientation)
			print(action)
		if done:

			self.learn()
			self.episode_reward = 0
			self.episode_timesteps = 0



		self.last_action = action
		self.last_state = new_state
		self.last_orientation = orientation
		self.last_reward = reward
		self.last_done = done


		self.episode_reward += reward
		self.episode_timesteps += 1

		return action

	def learn(self, batch_size=1024, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
		for it in tqdm(range(self.episode_timesteps)):
			# sampling a batch of transition (s, s', a, r) from the memory
			batch_states, batch_orientations, batch_next_states, batch_next_orientations, batch_actions, batch_rewards, batch_dones = self.replay_buffer.sample(batch_size)
			state = torch.Tensor(batch_states).to(device)
			orientation = torch.Tensor(batch_orientations).to(device)
			next_state = torch.Tensor(batch_next_states).to(device)
			next_orientation = torch.Tensor(batch_next_orientations).to(device)
			action = torch.Tensor(batch_actions).to(device)
			reward = torch.Tensor(batch_rewards).to(device)
			done = torch.Tensor(batch_dones).to(device)


			# getting next action a' from the next state s'
			next_action = self.actor_target(next_state, next_orientation)
			
			# adding gaussian noise to this next action a' 
			# and we clamp it in a range of values supported by the environment.
			noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
			noise = noise.clamp(-noise_clip, noise_clip)
			next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
			
			# two critic targets take each the couple(s', a') as input and return
			# the Q-values Qt1 and Qt2
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			
			# getting the minimum of above two Q values
			target_Q = torch.min(target_Q1, target_Q2)
			
			# getting the final target of two critic models
			target_Q = reward + (discount * target_Q * (1 - done)).detach()
			
			# two critic models take state s and action a to predict Q value,
			# which will be compared with the Q value derived from critic target
			current_Q1, current_Q2 = self.critic(state, action)
			
			# calculating the loss between Q value from critic model and critic target
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
			
			# updating the network parameters
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()
			
			# updating the actor model by gradient ascent andd it happens
			# once every two iterations.
			if it % policy_freq == 0:
				actor_loss = -self.critic.q1(state, self.actor(state, orientation)).mean()
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()
				
				# updating the weights of actor target and critic target
				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
					
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
