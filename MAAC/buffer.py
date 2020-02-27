import torch
from torch.autograd import Variable

import numpy as np




class ReplayBuffer:
    def __init__(self, capacity, num_agents, obs_dims, action_dims):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obsmemory = []
        self.actmemory = []
        self.rmemory = []
        self.nobsmemory = []
        self.donememory = []
        for _ in range(self.num_agents):
            self.obsmemory.append([])
            self.actmemory.append([])
            self.rmemory.append([])
            self.nobsmemory.append([])
            self.donememory.append([])
        self.filled = 0    
        self.current = 0

    def __len__(self):
        return self.filled_i
    
    def add(self, observations, actions, rewards, nobservations, dones):
        if self.filled < self.capacity:
            self.filled += 1
            self.current += 1
            for i in range(self.num_agents):
                self.obsmemory[i].append(observations[i,:])
                self.actmemory[i].append(actions[i])
                self.rmemory[i].append(rewards[i])
                self.nobsmemory[i].append(nobservations[i,:])
                self.donememory[i].append(dones[i])
            
        else: # when capacity is overblown, the buffer restarts storage from position 0
            for i in range(self.num_agents):
                self.obsmemory[i][self.current % self.capacity] = observations[i,:]
                self.actmemory[i][self.current % self.capacity] = actions[i]
                self.rmemory[i][self.current % self.capacity] = rewards[i]
                self.nobsmemory[i][self.current % self.capacity] = nobservations[i,:]
                self.donememory[i][self.current % self.capacity] = dones[i]
                self.current += 1
            
    def sample(self, batch_size, to_gpu=False):
        idxs = np.random.choice(np.arange(self.filled), size=batch_size, replace=False)
        device = 'cpu'
        if to_gpu:
            device = 'cuda'
        # normalizes rewards if required
        obs, actions, rewards, nobs, dones = [], [], [], [], []
        for i in range(self.num_agents):
            rewards.append(torch.FloatTensor(np.vstack([self.rmemory[i][idx] for idx in idxs])).to(device))
            obs.append(torch.FloatTensor(np.vstack([self.obsmemory[i][idx] for idx in idxs])).to(device))
            actions.append(torch.FloatTensor(np.vstack([self.actmemory[i][idx] for idx in idxs])).to(device))
            nobs.append(torch.FloatTensor(np.vstack([self.nobsmemory[i][idx] for idx in idxs])).to(device))
            dones.append(torch.ByteTensor(np.vstack([self.donememory[i][idx] for idx in idxs]).astype(np.uint8)).to(device))
            
        return (obs, actions, rewards, nobs, dones)
