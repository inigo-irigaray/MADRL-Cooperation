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
 
<p align=justify>Agent.py contains the main architecture of the algorithm where the core of deep learning structure is implemented. In particular, for the MADDPG case, this file designs an <b>Actor-Critic</b> method that learns policies from <b>centralized critics</b> that accumulate knowledge from other agents at training time, with each agent relying solely on its own individual policy at execution time. This allows agents to model their policy on expected behavior from other agents during training, so they can perform robustly in the environment when that information is lacking at test time. In particular, actors' architecture is similar to single-agent DDPG methods, but each agent has its own centralized critic estimating the mean Q-value of its value function after accumulating information from all other agents' actions.</p>

<p align=justify>The <b>Actor</b> class consists of an intial <b>Layer Normalization</b> layer of size 24, followed by three <b>fully-connected layers</b> of shape (24, 256) (256, 128) (128, 2), with two <b>Rectified Linear Unit</B> non-linearites in-between, and a final <b>Tanh</b> non-linearity as output. On the other hand, the <b>Critic</b> network consists of an initial <b>Layer Normalization</b> layer of size 48, followed by three <b>fully-connected layers</b> of shape (48, 256) ([256 + act_size*n_agents =] 260, 128) (128, 1), with two <b>ReLU</b> non-linearities in between.</p>

<p align=justify>The Target Model class allows for the clear and efficient creation of target actor and critic networks and handy hard or soft update functions. It creates a deep copy of the online models allowing for simple, periodical updates of their weights. This class, along the Actor and Critic class explained above, make up the heart of the <b>DDPGAgent</b> class. It instantiates indivual DDPG agents for the environment, calculates their next action based on their online actor with <b>random noise for exploration</b> (empirically tested to perform as optimally as Ornstein-Uhlenbeck noise, but much simpler, cleaner and easier to read), and allows it to save or set their networks' weights (handy for saving and initializing the environment from trained agents).</p>

<p align=justify>Finally, the MADDPG class instantiates all DDPG agents in the environment, allows them to act collectively and updates their networks parameters through backpropagation of their respective losses. Each online critic is updated through the backpropagation of a mean-squared error loss between the bootstrapped Q-values calculated by the target critic from all agents' next states and actions and the value calculated by the online critic based on all agents' current states and actions. Then, each online actor is updated with respect to its own estimated policy and its centralized online critic estimated Q-value from all agents' policies.</p>

<p align=justify>Additionally, this class contains utility functions to update all target networks, switch the model to CPUs and GPUs as required by different stages of the training process, saving the weights of a trained model, and initializing the environment from scratch or from a saved model.</p>

### buffer.py

<p align=justify>This file contains the basic experience replay buffer to store states, actions, rewards, next_states and dones from agents' interactions with the environment and allow them to be sampled randomly for training and learning off-policy.</p>

### main.py

<p align=justify>Finally, main.py contains the core to run the algorithm. It allows us to automatically run a function by calling main.py from the command line with our preferred hyperparameters that will initialize training of our agents and save the models' weights and parameters once they have completed the goal of achieving >0.5 reward  on average for the past 100 episodes.</p>

## Plot of Rewards

<p align=justify>The plot below shows episodic rewards of the agents, calculated as the maximum of the two agents' rewards for the episode. The environment was solved in 770 episodes, with longer, more consistent and successful games after episode 600. The agents achieved a maximum episodic reward of 2.5 before training was halted with goal completion. This seems pretty impressive after the consistently poor performance of both agents for the first couple hundred of episodes.</p>

<p align="center"><img src="https://github.com/inigo-irigaray/MADRL-Cooperation/blob/master/imgs/agents-multi-rewards.png"></p>

<p align="center"><img src="https://github.com/inigo-irigaray/MADRL-Cooperation/blob/master/imgs/smoothed_reward.png"></p>

## Future Work

<p align=justify>Future work revolves around finalizing all the tweaks on the attention MAAC implementation, and testing both models on the soccer competitive environment.</p>
