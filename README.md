# MADRL-Cooperation

## Tennis Cooperative Environment

<p align=justify>In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.</p>

<p align=justify>The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.</p>

<p align=justify>The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,</p>

<p align=justify> • After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores.</p>
<p align=justify> • We then take the maximum of these 2 scores. This yields a single score for each episode.</p>

<p align=justify>The task is solved, when the average (over 100 episodes) of those scores is at least +0.5.</p>

<p align=justify>The environment can be installed from the following links:</p>
 
 · <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip>Linux</a>
 
 · <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip>MacOS</a>
 
 · <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip>Windows(32-bit)</a>
 
 · <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip>Windows(64-bit)</a>
 
 · <a href=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip>AWS(headless)</a>
 
 ### Requirements
 
 Nanodegree's prerequisites: <a href=https://github.com/udacity/deep-reinforcement-learning/#dependencies>link.</a>
 
    python==3.6
    tensorflow==1.7.1
    Pillow>=4.2.1
    matplotlib
    numpy>=1.11.0
    jupyter
    pytest>=3.2.2
    docopt
    pyyaml
    protobuf==3.5.2
    grpcio==1.11.0
    torch==0.4.0
    pandas
    scipy
    ipykernel
    tensorboardX==1.4
    unityagents

### Running the models

<p align=justify> To run the different models available in this repository one only needs to clone/download from this repository the appropiate files from the folder of the model he/she wants to run and write in the command line: </p>

    $ python main.py
or

    $ python3 main.py
    
<p align=justify> which will start training the model from scratch until it reaches the environment's goal.</p>

<p align=justify>For example:
<p align=justify>1. Clone this repository.</p>
<p align=justify>2. Install all required dependencies.</p>
<p align=justify>3. And run the the command:</p>
     
    $ python3 path_to_MADDPG_folder/main.py --cuda=True
