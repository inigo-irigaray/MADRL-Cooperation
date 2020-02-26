import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from itertools import chain




class Actor(nn.Module):
    def __init__(self, obs_size, act_size, hid1=400, hid2=300, norm='layer', onehot_dim=0):
        super(Actor, self).__init__()
        if norm=='batch':
            self.inp = nn.BatchNorm1d(obs_size)
        elif norm=='layer':
            self.inp = nn.LayerNorm(obs_size)
        else:
            self.inp = lambda x: x
        
        self.net = nn.Sequential()
        self.net.add_module('fc1', nn.Linear(obs_size + onehot_dim, hid1))
        self.net.add_module('nl1', nn.ReLU())
        self.net.add_module('fc2', nn.Linear(hid1, hid2))
        self.net.add_module('nl2', nn.ReLU())
        self.net.add_module('fc3', nn.Linear(hid2, act_size))
        
    def forward(self, x): #sample=True
        one_hot = None
        
        # separates one_hot to avoid batch normalizing it
        if type(x) is tuple:
            x, one_hot = x
            
        # normalizes observation input if required
        inp = self.inp(x)
        if one_hot is not None:
            inp = torch.cat((inp, one_hot), dim=1)
        out = self.net(inp) 
        actions = out
        regularizer = (out ** 2).mean() # attention heads' regularizers

        return actions, regularizer
    
    
    
    
class Encoder(nn.Module):
    def __init__(self, inp, hidenc, norm='layer'):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential()
        if norm=='layer':
            self.encoder.add_module('ln1', nn.LayerNorm(inp, elementwise_affine=False))
        elif nornm=='batch':
            self.encoder.add_module('bn1', nn.BatchNorm1d(inp, affine=False))
        else:
            pass
        
        self.encoder.add_module('fc1', nn.Linear(inp, hidenc))
        self.encoder.add_module('nl1', nn.LeakyReLU())
     
    
    def forward(self, x):
        return self.encoder(x)
    
    
    
    
class Critic(nn.Module):
    def __init__(self, inp, outp, hidc, norm='layer'):
        super(Critic, self).__init__()
        self.critic = nn.Sequential()
        if norm=='layer':
            self.critic.add_module('ln1', nn.LayerNorm(inp, elementwise_affine=False))
        elif norm=='batch':
            self.critic.add_module('bn1', nn.BatchNorm1d(inp, affine=False))
        else:
            pass
        
        self.critic.add_module('fc1', nn.Linear(inp, hidc))
        self.critic.add_module('nl1', nn.LeakyReLU())
        self.critic.add_module('fc2', nn.Linear(hidc, outp))
        
    def forward(self, x):
        return self.critic(x)
        
        
        
        
class AttentionCritic(nn.Module):
    def __init__(self, sa_size, hidenc=400, hidc=400, norm='layer', att_heads=1):
        super(AttentionCritic, self).__init__()
        self.sa_size = sa_size
        self.n_agents = len(sa_size)
        self.att_heads = att_heads
        
        # create state & state-action pairs encoders, and critics for each agent
        self.s_encoders = nn.ModuleList()
        self.sa_encoders = nn.ModuleList() 
        self.critics = nn.ModuleList()
        for state_dim, act_dim in sa_size:
            self.s_encoders.append(Encoder(state_dim, hidc, norm))
            self.sa_encoders.append(Encoder(state_dim+act_dim, hidc, norm))
            self.critics.append(Critic(hidc*2, act_dim, hidc, norm))
        
        # creates keys & values extractors, and agent selectors for each attention head
        att_dim = hidc // att_heads
        self.key_feats = nn.ModuleList() # where the model should attend to
        self.value_feats = nn.ModuleList() # features for state-action value estimation
        self.agent_selector = nn.ModuleList() # selects which agents each individual agent should pay more attention to
        for i in range(att_heads):
            self.key_feats.append(nn.Linear(hidc, att_dim, bias=False))
            self.value_feats.append(nn.Linear(hidc, att_dim, bias=False)) 
            self.agent_selector.append(nn.Sequential(nn.Linear(hidc, att_dim), nn.LeakyReLU()))
            
        self.shared_modules = [self.key_feats, self.value_feats, self.agent_selector, self.sa_encoders]
        
    def shared_parameters(self):
        return chain(*[module.parameters() for module in self.shared_modules])
    
    def scale_shared_grads(self):
        for parameter in self.shared_parameters():
            if parameter.grad is not None:
                parameter.grad.data.mul_(1. / self.n_agents)
            #print(parameter.shape)
            #if parameter.grad is None:
                #print(parameter.grad)
    
    def forward(self, x, agents=None, return_q=True, return_allqs=False, regularize=False, return_att=False):
        if agents is None:
            agents = range(len(self.sa_encoders))
        
        #print(self.s_encoders[0][0])
        # creates input types list of size 'agents' with tensors for each input for each agent    
        states = [state for state, _ in x] # list with tensors of shape (batch_size, observation size)
        actions = [action for _, action in x] # list with tensors of shape (batch_size, action size)
        state_actions = [torch.cat((state, action), dim=1) for state, action in x] # l w/ tensors of shape (batch, obs+act_size)
        
        # creates lists of state & state-actions encodings of length 'agents' with tensors for each encoding
        s_encodings = [self.s_encoders[agent](states[agent]) for agent in agents] # list with tensors of shape (batch, hidc)     
        sa_encodings = [enc(sa) for enc, sa in zip(self.sa_encoders, state_actions)] # list w/ tensors of shape (batch, hidc)
        
        # creates list(len'heads') of lists(len'agents') of keys, values and agent selectors for each head & agent
        ## w/ tensors of shape(batch, att_dim)
        heads_keys = [[k_extractor(saenc) for saenc in sa_encodings] for k_extractor in self.key_feats]
        heads_vals = [[v_extractor(saenc) for saenc in sa_encodings] for v_extractor in self.value_feats]
        heads_selectors = [[sel_extractor(saenc) for saenc in sa_encodings] for sel_extractor in self.agent_selector]
        
        # creates list(len'agents') of lists for all values, attention logits and attention probs for each agent
        all_values = [[] for _ in range(len(agents))]
        all_att_logits = [[] for _ in range(len(agents))]
        all_att_probs = [[] for _ in range(len(agents))]
        ## iterates over heads in keys, vals and selectors lists
        for h_keys, h_vals, h_selectors in zip(heads_keys, heads_vals, heads_selectors):
            ### iterates over agents within each head in the keys, vals and selectors lists
            for i, agent, selector in zip(range(len(agents)), agents, h_selectors):
                # creates list(len'agents-1') for the keys & vals of all agents other than the current agent being iterated over
                keys = [key for j, key in enumerate(h_keys) if j != agent] # list w/ tensors of shape (batch, att_dim)
                vals = [val for j, val in enumerate(h_vals) if j != agent] # list w/ tensors of shape (batch, att_dim)
                
                # calculates Attentive Communication as per formula (15) in 'Multi-Focus Attention Network for Efficient DRL'
                keys_transpose = torch.stack(keys).permute(1, 2, 0) # transpose keys, (batch, att_dim, agents-1)
                selector = selector.view(selector.shape[0], 1, -1) # (batch, 1, att_dim) for matrix multiplication
                att_logits = torch.matmul(selector, keys_transpose) # (batch, 1, agents-1)
                scaled_logits = att_logits / np.sqrt(keys[0].shape[1])
                att_weights = F.softmax(scaled_logits, dim=2) # (batch, 1, agents-1)
                
                # calculates q-values estimators as per formulas (16) in 'Multi-Focus Attention Network for Efficient DRL'
                vals_transpose = torch.stack(vals).permute(1, 2, 0) # for dot-product, (batch, att_dim, agents-1)
                qvals = (vals_transpose * att_weights).sum(dim=2) # (batch, att_dim, agents-1)
                
                all_values[i].append(qvals)
                all_att_logits[i].append(att_logits)
                all_att_probs[i].append(att_weights)
                
        # 
        rets = []
        for i, agent in enumerate(agents):
            agent_rets = []
            all_qs = self.critics[agent](torch.cat((s_encodings[i], *all_values[i]), dim=1))
            actions_idx = actions[agent].max(dim=1, keepdim=True)[1]
            q = all_qs.gather(1, actions_idx)
            if return_q:
                agent_rets.append(q)
            if return_allqs:
                agent_rets.append(all_qs)
            if regularize:
                att_reg = 1e-3 * sum((logit**2).mean() for logit in all_att_logits[i])
                regs = (att_reg,)
                agent_rets.append(regs)
            if return_att:
                agent_rets.append(np.array(all_att_probs[i]))
            if len(agent_rets)==1:
                rets.append(agent_rets[0])
            else:
                rets.append(agent_rets)
        
        if len(rets)==1:
            return rets[0]
        return rets
                


        
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
    
    
    
    
class AttentionAgent:
    def __init__(self, obs_size, act_size, hid1=400, hid2=300, norm='layer', onehot_dim=0, lra=0.01):
        self.actor = Actor(obs_size, act_size, hid1=hid1, hid2=hid2, norm=norm, onehot_dim=onehot_dim)
        self.tgt_actor = TargetModel(self.actor)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lra)
        self.epsilon = 0.3
        
    def step(self, obs, explore=True):
        actions, regularizer = self.actor(obs)
        actions = actions.data.cpu().numpy()
        if explore:
            actions += self.epsilon * np.random.normal(size=actions.shape)
        actions = np.clip(actions, -1, 1)
        return actions, regularizer
        
    def get_params(self):
        return {'actor': self.actor.state_dict(),
                'target_actor': self.tgt_actor.tgt_model.state_dict(),
                'actor_optimizer': self.actor_optim.state_dict()}
        
    def set_params(self, params):
        self.actor.load_state_dict(params['actor'])
        self.tgt_actor.tgt_model.load_state_dict(params['target_actor'])
        self.actor_optim.load_state_dict(params['actor_optimizer'])
        
        
        
        
class AttentionSAC:
    def __init__(self, init_params, sa_size, norm='layer', gamma=0.99, tau=0.01, lra=0.01, lrc=0.01, 
                hid1=400, hid2=300, hidc=400, att_heads=1, **kwargs):
        self.n_agents = len(sa_size)
        self.agents = [AttentionAgent(hid1=hid1, hid2=hid2, norm=norm, lra=lra, **params) for params in init_params]
        self.critic = AttentionCritic(sa_size, hidc=hidc, norm=norm, att_heads=att_heads)
        self.tgt_critic = TargetModel(self.critic)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lrc, weight_decay=1e-3)
        
        self.init_params = init_params
        self.gamma = gamma
        self.tau = tau
        self.lra = lra
        self.lrc = lrc
        self.idx = 0
        
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
        return [agent.step(obs, explore)[0] for agent, obs in zip(self.agents, observations)],\
               [agent.step(obs, explore)[1] for agent, obs in zip(self.agents, observations)]
    
    def update_critic(self, sample, writer=None, **kwargs):
        obs, actions, rewards, next_obs, dones = sample
        next_actions = []
        for actor, nob in zip(self.tgt_actors, next_obs):
            current_next_action, _ = actor(nob)
            next_actions.append(current_next_action)
        next_qvals = self.tgt_critic.tgt_model(list(zip(next_obs, next_actions)))
        
        outp_critics = self.critic(list(zip(next_obs, next_actions)), regularize=True)
        
        q_loss = 0
        for agent_i, next_q, (estimate_q, regularizers) in zip(range(self.n_agents), next_qvals, outp_critics):
            next_q[dones[agent_i]] = 0.0
            tgt_q = (rewards[agent_i] + self.gamma * next_q)
            q_loss += F.mse_loss(next_q, tgt_q.detach())
            for regularizer in regularizers:
                q_loss += regularizer
                
        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = nn.utils.clip_grad_norm(self.critic.parameters(), 10 * self.n_agents)
        self.critic_optim.step()
        self.critic_optim.zero_grad()
        self.idx += 1
        
        if writer is not None:
            writer.add_scalar('losses/q_loss', q_loss, self.idx)
            writer.add_scalar('grad_norms/q', grad_norm, self.idx)
            
        
    def update_actors(self, sample, writer=None, **kwargs):
        obs, actions, rewards, next_obs, dones = sample
        sample_actions = []
        all_logits = []
        all_probs = []
        all_regularizers = []
        
        for agent_i, actor, ob in zip(range(self.n_agents), self.actors, obs):
            current_action, regularizer = actor(ob)
            sample_actions.append(current_action)
            all_regularizers.append(regularizer)
            
        qs, all_qs = self.critic(list(zip(obs, sample_actions)), return_allqs=True)
        for agent_i, regularizers, qval, all_qvals in zip(range(self.n_agents), all_probs, all_logits, 
                                                                         all_regularizers, qs, all_qs):
            current_agent = self.agents[agent_i]
            value = (all_qvals * probs).sum(dim=1, keepdim=True)
            actor_target = qval - value
            actor_loss = (logits * (-actor_target).detach()).mean()
            
            for regularizer in regularizers:
                actor_loss += 1e-3 * regularizer
            
            self.critic.disable_grads()
            actor_loss.backward()
            self.critic.enable_grads()
            
            gard_norm = nn.utils.clip_grad_norm(current_agent.actor.parameters(), 0.5)
            current_agent.actor_optim.step()
            current_agent.actor_optim.zero_grad()
            
            if writer is not None:
                writer.add_scalar('agent_%i/losses/pol_loss' % agent_i, actor_loss, self.idx)
                writer.add_scalar('agent_%i/grad_norms/actor' % agent_i, grad_norm, self.idx)
            
    def update_all_targets(self):
        self.tgt_critic.soft_update(self.tau)
        for agent in self.agents:
            agent.tgt_actor.soft_update(self.tau)
            
    def prep_training(self, device='gpu'):
        self.critic.train()
        self.tgt_critic.tgt_model.train()
        for agent in self.agents:
            agent.actor.train()
            agent.tgt_actor.tgt_model.train()
        
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
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.tgt_critic_dev == device:
            self.tgt_critic.tgt_model = fn(self.tgt_critic.tgt_model)
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
                     'agent_params': [agent.get_params() for agent in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'tgt_critic': self.tgt_critic.tgt_model.state_dict(),
                                       'critic_optim': self.critic_optim.state_dict()}}
        torch.save(save_dict, filename)
        
    @classmethod
    def init_from_env(cls, env_info, brain, norm='layer', gamma=0.99, tau=0.01, lra=0.01,
                      lrc=0.01, hid1=400, hid2=300, hidc=400, att_heads=4, **kwargs):
        init_params = []
        n_agents = len(env_info.agents)
        sa_size = []
        for _ in range(n_agents): # creates init_params for each agent's future instantiation
            obs_size = env_info.vector_observations.shape[1]
            act_size = brain.vector_action_space_size
            sa_size.append((obs_size, act_size))
            
            init_params.append({'obs_size': obs_size,
                                'act_size': act_size})
            
        init_dict = {'norm': norm,
                     'gamma': gamma,
                     'tau': tau,
                     'lra': lra,
                     'lrc': lrc,
                     'hid1': hid1,
                     'hid2': hid2,
                     'hidc': hidc,
                     'att_heads': att_heads,
                     'init_params': init_params,
                     'sa_size': sa_size
                    }
        
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        
        return instance
    
    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        save_dict = torch.load(filename)
        instance = cls(**dave_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for agent, parameters in zip(instance.agents, save_dict['agent_params']):
            agent.set_params(parameters)
            
        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.tgt_critic.tgt_model.load_state_dict(critic_params['tgt_critic'])
            instance.critic_optim.load_state_dict(critic_params['critic_optim'])
            
        return instance
