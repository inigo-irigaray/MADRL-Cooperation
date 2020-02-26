import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np




class Actor(nn.Module):
    def __init__(self, obs_size, act_size, hid1=400, hid2=300, norm='layer'):
        super(Actor, self).__init__()
        self.net = nn.Sequential()
        if norm=='batch':
            self.net.add_module('bn1', nn.BatchNorm1d(obs_size))
        elif norm=='layer':
            self.net.add_module('ln1', nn.LayerNorm(obs_size))
        else:
            pass
        self.net.add_module('fc1', nn.Linear(obs_size, hid1))
        self.net.add_module('nl1', nn.ReLU())
        self.net.add_module('fc2', nn.Linear(hid1, hid2))
        self.net.add_module('nl2', nn.ReLU())
        self.net.add_module('fc3', nn.Linear(hid2, act_size))
        self.net.add_module('nl3', nn.Tanh())
        
    def forward(self, x):
        return self.net(x)

    
    
        
class Critic(nn.Module):
    def __init__(self, obs_size, act_size, num_agents, hid1=400, hid2=300, norm='layer'):
        super(Critic, self).__init__()
        self.net1 = nn.Sequential()
        if norm=='batch':
            self.net1.add_module('bn1', nn.BatchNorm1d(obs_size*num_agents))
        elif norm=='layer':
            self.net1.add_module('ln1', nn.LayerNorm(obs_size*num_agents))
        else:
            pass
        self.net1.add_module('fc1', nn.Linear(obs_size*num_agents, hid1))
        self.net2 = nn.Sequential()
        self.net2.add_module('nl1', nn.ReLU())
        self.net2.add_module('fc2', nn.Linear(hid1 + act_size*num_agents, hid2))
        self.net2.add_module('nl2', nn.ReLU())
        self.net2.add_module('fc3', nn.Linear(hid2, 1))
                
    def forward(self, x, a):
        o = self.net1(x)
        return self.net2(torch.cat((o, a), dim=1))
    
    
    
    
class TargetModel:
    def __init__(self, model):
        self.model = model
        self.tgt_model = copy.deepcopy(self.model)
        
    def hard_update(self):
        self.tgt_model.load_state_dict(self.model.state_dict())
    
    def soft_update(self, tau):
        assert isinstance(tau, float)
        assert 0. < tau <= 1.
        state = self.model.state_dict()
        tgt_state = self.tgt_model.state_dict()
        for key, value in state.items():
            tgt_state[key] = tgt_state[key] * (1 - tau) + value * tau
        self.tgt_model.load_state_dict(tgt_state)

        
        
        
class DDPGAgent:
    def __init__(self, obs_size, act_size, num_agents, hid1=400, hid2=300, norm='layer', lra=1e-4, lrc=1e-3, epsilon=0.3): 
        self.actor = Actor(obs_size, act_size, hid1=hid1, hid2=hid2, norm=norm)
        self.critic = Critic(obs_size, act_size, num_agents, hid1=hid1, hid2=hid2, norm=norm)
        self.tgt_actor = TargetModel(self.actor)
        self.tgt_critic = TargetModel(self.critic)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lra)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lrc)
        self.epsilon = epsilon
        self.obs_size = obs_size
        self.act_size = act_size
        self.num_agents = num_agents
        
    def step(self, obs, explore=False):
        action = self.actor(obs)
        action = action.data.cpu().numpy()
        if explore:
            action += self.epsilon * np.random.normal(size=action.shape)
        action = np.clip(action, -1, 1)
        return action
    
    def get_params(self):
        return {'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'tgt_actor': self.tgt_actor.tgt_model.state_dict(),
                'tgt_critic': self.tgt_critic.tgt_model.state_dict(),
                'actor_optim': self.actor_optim.state_dict(),
                'critic_optim': self.critic_optim.state_dict()}
    
    def set_params(self, params):
        self.actor.load_state_dict(params['actor'])
        self.critic.load_state_dict(params['critic'])
        self.tgt_actor.tgt_model.load_state_dict(params['tgt_actor'])
        self.tgt_critic.tgt_model.load_state_dict(params['tgt_critic'])
        self.actor_optim.load_state_dict(params['actor_optim'])
        self.critic_optim.load_state_dict(params['critic_optim'])
        
        
        
        
class MADDPG:
    def __init__(self, init_params, n_agents, hid1=400, hid2=300, norm='layer',
                 lra=1e-4, lrc=1e-3, epsilon=0.3, gamma=0.99, tau=0.01):
        self.n_agents = n_agents
        self.agents = [DDPGAgent(hid1=hid1, hid2=hid2, norm=norm, lra=lra, lrc=lrc, epsilon=epsilon, **params)
                       for params in init_params]
        self.init_params = init_params
        self.gamma = gamma
        self.tau = tau
        self.iter = 0
        self.act_dev = 'cpu'
        self.tgt_act_dev = 'cpu'
        self.critic_dev = 'cpu'
        self.tgt_critic_dev = 'cpu'
        
    @property
    def actors(self):
        return [agent.actor for agent in self.agents]
    
    @property
    def tgt_actors(self):
        return [agent.tgt_actor.tgt_model for agent in self.agents]
    
    def step(self, observations, explore=False):
        return [agent.step(obs, explore=explore) for agent, obs in zip(self.agents, observations)]
    
    def update(self, sample, agent_i, writer):
        obs, actions, rewards, next_obs, dones = sample
        current_agent = self.agents[agent_i]
        
        # train critic
        
        current_agent.critic_optim.zero_grad()
        next_actions = [act_i(nobs) for act_i, nobs in zip(self.tgt_actors, next_obs)]# self.tgt_actors @property(check above)
        ## 
        next_actions = torch.stack(next_actions).permute(1, 0, 2).contiguous()
        next_actions = next_actions.view(-1, self.agents[agent_i].act_size*self.agents[agent_i].num_agents)
        ## 
        next_obs_v = torch.stack(next_obs).permute(1, 0, 2).contiguous()
        next_obs_v = next_obs_v.view(-1, self.agents[agent_i].obs_size*self.agents[agent_i].num_agents)
        ## 
        next_val = current_agent.tgt_critic.tgt_model(next_obs_v, next_actions)
        next_val[dones[agent_i]] = 0.0
        tgt_value = rewards[agent_i] + self.gamma * next_val
        ## 
        obs_v = torch.stack(obs).permute(1, 0, 2).contiguous()
        obs_v = obs_v.view(-1, self.agents[agent_i].obs_size*self.agents[agent_i].num_agents)
        ## 
        actions_v = torch.stack(actions).permute(1, 0, 2).contiguous()
        actions_v = actions_v.view(-1, self.agents[agent_i].act_size*self.agents[agent_i].num_agents)
        ## 
        value = current_agent.critic(obs_v, actions_v)
        critic_loss = F.mse_loss(value, tgt_value.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(current_agent.critic.parameters(), 0.5)
        current_agent.critic_optim.step()
        
        # train actor
        
        current_agent.actor_optim.zero_grad()
        ## 
        current_act_out = current_agent.actor(obs[agent_i])
        all_actor_actions = []
        for i, actor_i, ob in zip(range(self.n_agents), self.actors , obs):# self.actors @property(check above)
            if i == agent_i:
                all_actor_actions.append(current_act_out)
            else:
                all_actor_actions.append(actor_i(ob))
        ## 
        all_actor_actions = torch.stack(all_actor_actions).permute(1, 0, 2).contiguous()
        all_actor_actions = all_actor_actions.view(-1, self.agents[agent_i].act_size*self.agents[agent_i].num_agents)
        ## 
        actor_loss = -current_agent.critic(obs_v, all_actor_actions).mean()
        actor_loss += (current_act_out ** 2).mean() * 1e-3
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm(current_agent.actor.parameters(), 0.5)
        current_agent.actor_optim.step()
        
        writer.add_scalars('agent%i/losses' % agent_i, {'critic_loss': critic_loss, 'actor_loss': actor_loss}, self.iter)
        
    def update_all_targets(self, soft=True):
        for agent in self.agents:
            if soft:
                agent.tgt_actor.soft_update(self.tau)
                agent.tgt_critic.soft_update(self.tau)
            else:
                agent.tgt_actor.hard_update()
                agent.tgt_critic.hard_update()
        self.iter += 1
        
    def prep_training(self, device='gpu'):
        for agent in self.agents:
            agent.actor.train()
            agent.critic.train()
            agent.tgt_actor.tgt_model.train()
            agent.tgt_critic.tgt_model.train()
        
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        
        if not self.act_dev == device:
            for agent in self.agents:
                agent.actor = fn(agent.actor)
            self.act_dev = device
        if not self.tgt_act_dev == device:
            for agent in self.agents:
                agent.tgt_actor.tgt_model = fn(agent.tgt_actor.tgt_model)
            self.tgt_act_dev = device
        if not self.critic_dev == device:
            for agent in self.agents:
                agent.critic = fn(agent.critic)
            self.critic_dev = device
        if not self.tgt_critic_dev == device:
            for agent in self.agents:
                agent.tgt_critic.tgt_model = fn(agent.tgt_critic.tgt_model)
            self.tgt_critic_dev = device
            
    def prep_rollouts(self, device='cpu'):
        for agent in self.agents: # only need actors' policies for rollouts
            agent.actor.eval()
        if device=='gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.act_dev == device:
            for agent in self.agents:
                agent.actor = fn(agent.actor)
            self.act_dev = device
        
    def save(self, filename):
        self.prep_training(device='cpu') # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [agent.get_params() for agent in self.agents]}
        torch.save(save_dict, filename)
        
    @classmethod
    def init_from_env(cls, env_info, brain, hid1=400, hid2=300, norm='layer',
                      lra=1e-4, lrc=1e-3, epsilon=0.3, gamma=0.99, tau=0.01):
        init_params = []
        n_agents = len(env_info.agents)
        for _ in range(n_agents): # creates init_params for each agent's future instantiation
            obspace = env_info.vector_observations.shape[1]
            aspace = brain.vector_action_space_size
            # get actor params
            in_act = obspace
            out_act = aspace
            # get critic params
            in_crt = 0
            for _ in range(n_agents):
                in_crt += obspace
                in_crt += aspace
            
            init_params.append({'obs_size': in_act,
                                'act_size': out_act,
                                'num_agents': n_agents})
            
        init_dict = {'init_params': init_params,
                     'n_agents': n_agents,
                     'hid1': hid1,
                     'hid2': hid2,
                     'norm': norm,
                     'lra': lra,
                     'lrc': lrc,
                     'epsilon': epsilon,
                     'gamma': gamma,
                     'tau': tau}
        
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        
        return instance
    
    @classmethod
    def init_from_save(cls, filename):
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for agent, params in zip(instance.agents, save_dict['agent_params']):
            agent.set_params(params)
            
        return instance
