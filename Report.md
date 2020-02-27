# DDPG Implementation for Multi-Agent Environments (MADDPG)

## Hyperparameters

    --env = "/data/Tennis_Linux_NoVis/Tennis"
    --cuda = True
    --seed = 1
    --hid1 = 256
    --hid2 = 128
    --norm = 'layer'
    --lra = 1e-4
    --lrc = 1e-3
    --epsilon = 0.3
    --gamma = 0.99
    --tau = 6e-2
    --capacity = 100000
    --batch_size = 128

## Algorithms

<p align=justify>In this work I implement a multi-agent variant of DDPG, where agents learn a target policy from an online behaviour policy interacting with the environment and other agents.</p>

### agent.py

<p align=justify>Agent.py contains the main architecture of the algorithm where the core of deep learning structure is implemented. In particular, for the MADDPG case, this file designs an Actor-Critic method that learns policies from centralized critics that accumulate knowledge from other agents at training time, with each agent relying solely on its own individual policy at execution time. This allows agents to model their policy on expected behavior from other agents during training, so they can perform robustly in the environment when that information is lacking at test time. In particular, actors' architecture is similar to single-agent DDPG methods, but each agent has its own centralized critic estimating the mean Q-value of its value function after accumulating information from all other agents' actions.</p>

<p align=justify>The Actor class consists of an intial Layer Normalization layer of size 24, followed by three fully-connected layers of shape (24, 256) (256, 128) (128, 2), with two Rectified Linear Unit non-linearites in-between, and a final Tanh non-linearity as output. On the other hand, the Critic network consists of an initial Layer Normalization layer of size 48, followed by three fully-connected layers of shape (48, 256) ([256 + act_size*n_agents =] 260, 128) (128, 1), with two ReLU non-linearities in between.</p>

<p align=justify>The Target Model class allows for the clear and efficient creation of target actor and critic networks and handy hard or soft update functions. It creates a deep copy of the online models allowing for simple, periodical updates of their weights. This class, along the Actor and Critic class explained above, make up the heart of the DDPGAgent class. It instantiates indivual DDPG agents for the environment, calculates their next action based on their online actor with random noise for exploration (empirically tested to perform as optimally as Ornstein-Uhlenbeck noise, but much simpler, cleaner and easier to read), and allows it to save or set their networks' weights (handy for saving and initializing the environment from trained agents).</p>

<p align=justify>Finally, the MADDPG class 
