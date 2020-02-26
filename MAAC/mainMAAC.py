import argparse
import os
import time
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import numpy as np

from pathlib import Path
from unityagents import UnityEnvironment
from collections import deque

import buffer
import MAACAgent




def run(config):
    model_dir = Path('./MAAC/')
    if not model_dir.exists():
        current_run = 'run1'
    else:
        run_nums = [int(str(folder.name).split('run')[1]) 
                        for folder in model_dir.iterdir() if str(folder.name).startswith('run')]
        if len(run_nums) == 0:
            current_run = 'run1'
        else:
            current_run = 'run%i' % (max(run_nums) + 1)
            
    run_dir = model_dir / current_run
    logs_dir = run_dir / 'logs'
    os.makedirs(logs_dir)
    
    writer = SummaryWriter(str(logs_dir))
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    if torch.cuda.is_available() and config.cuda==True:
        cuda = True
    else:
        cuda = False
    env = UnityEnvironment(file_name="/data/Tennis_Linux_NoVis/Tennis")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    
    maac = MAACAgent.AttentionSAC.init_from_env(env_info, brain, norm=config.norm, gamma=config.gamma, tau=config.tau,
                                                lra=config.lra, lrc=config.lrc, hid1=config.hid1, hid2=config.hid2,
                                                hidc=config.hidc, att_heads=config.att_heads)
    
    repbuffer = buffer.ReplayBuffer(config.capacity, maac.n_agents,
                                 [brain.vector_observation_space_size for _ in range(maac.n_agents)],
                                 [brain.vector_action_space_size for _ in range(maac.n_agents)])
    
    for i, agent in enumerate(maac.agents):
        print('\nAgent %i:\n' % i)
        print(agent.actor)
    print('\n', maac.critic)
    
    episode = 0
    rewards_100 = deque(maxlen=100)
    while True:
        t = time.time()
        total_rewards = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        obs = env_info.vector_observations
        maac.prep_rollouts(device='cpu')
        
        while True:
            obs_v = [Variable(torch.Tensor(obs[agent_i, :]), requires_grad=False) 
                     for agent_i in range(maac.n_agents)]
            actions, regularizer = maac.step(obs_v, explore=True) #double check DDPG.step FloatTensor part instead of Variable
            env_info = env.step(actions)[brain_name]
            next_obs = env_info.vector_observations
            rewards = env_info.rewards
            total_rewards += rewards
            dones = env_info.local_done

            repbuffer.add(obs, actions, rewards, next_obs, dones)
            if np.any(dones):
                episode_reward = np.max(total_rewards)
                rewards_100.append(episode_reward)
                writer.add_scalar('episode_reward', episode_reward, episode)
                print("\n\nDone episode %d for an episode reward of %.3f in %.2f seconds."
                      % (episode, episode_reward, (time.time() - t)))
                t = time.time()
                break
            
            obs = next_obs
            if repbuffer.filled > config.batch_size:
                if cuda:
                    maac.prep_training(device='gpu')
                else:
                    maac.prep_training(device='cpu')
                
                sample = repbuffer.sample(config.batch_size, to_gpu=cuda)
                maac.update_critic(sample, writer=writer)
                maac.update_actors(sample, writer=writer)
                maac.update_all_targets()
                maac.prep_rollouts(device='cpu')
                
        episode += 1
        for agent_i, r in enumerate(total_rewards):
            writer.add_scalar('agent%i-episode_rewards' % agent_i, r, episode)
            print('Agent %i: episode reward of %.2f.' % (agent_i, r))
            
        if np.mean(rewards_100) > 0.5:
            print("Solved the environment in %i episodes!" % episode)
            break
            
    maac.save(run_dir / 'tennisMAAC.pt')
    env.close()
    writer.export_scalars_to_json(str(logs_dir / 'summary.json'))
    writer.close()    
    

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, type=bool)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--hid1', default=256, type=int)
    parser.add_argument('--hid2', default=128, type=int)
    parser.add_argument('--hidc', default=256, type=int)
    parser.add_argument('--norm', default='layer', type=str, help="Normalization layer takes values 'batch' for BatchNorm, \
                                                                   'layer' for LayerNorm, and any other value for no \
                                                                    normalization layer.")
    parser.add_argument('--lra', default=1e-4, type=float)
    parser.add_argument('--lrc', default=1e-3, type=float)
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--tau', default=6e-2, type=float)
    parser.add_argument('--capacity', default=100000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--att_heads', default=4, type=int)

    config = parser.parse_args()
    
    run(config)
