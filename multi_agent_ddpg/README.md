## Important Consideration  

1. To run the code, please use the command "./run_training.sh". The bash script cleans up and DELETE previous runs. The script is necessary because we need an extra command to ensure image rendering is possible remotely. Training takes about two hour. If you run locally on your own computer. Be sure to increase the number of parallel agents to the number of cores your computer have in main.py. GPU does not help that much in the computation.

2. To see a visualization of the results, run the script "./run_tensorboard.sh". A link will appear, and direct your browser to that link to see rewards over time and other statistics

3. The trained models are stored in "model_dir" by default. You can also find .gif animations that show how the agents are performing! The gif file contains a grid of separate parallel agents.

4. To understand the goal of the environment: blue dots are the "good agents", and the Red dot is an "adversary". All of the agents' goals are to go near the green target. The blue agents know which one is green, but the Red agent is color-blind and does not know which target is green/black! The optimal solution is for the red agent to chase one of the blue agent, and for the blue agents to split up and go toward each of the target.

5. it seems from code Agent 0 is the advesary and agent 1 and 2 collaborate.  

## Setup Environment

follow the instruction in the .yml file  conda env create --file environment.yml --prefix=/media/Data/env/maddpg  


## Conceptual Considerations

In MADDPG, each agent’s critic is trained using the observations and actions from all the agents, whereas each agent’s actor is trained using just its own observations. There are two types of observations one is the observation state space of all the agents in the 4 parallel env. for the Physical Dception environment with three agents it is of dimension 3x14. obs_full is world state irrespective of the agents, i.e., god eye view fron the goal perspective, and its dimension is 14. Determine which one is used for training the critic model. In original MaDDPG, input to critic network is all agents’ state and all agents’ action concatenated into one vector. Instead we divide this input into state (for keeping the environment stationary for that particular agent), which includes state of all agents and actions of all OTHER agents except the acting agent. These all serve as input to NN in first layer. Now acting agent action is fead into NN only after first layer to get the Q-Value for this (state, action) pair. This stabilises the agent greatly with all other parameters kept same.


## DDPG Implementation 
DDPG (Project2) as my base and tried 2 agents with common Replay buffer and Actor - Critic Network. But the issue is - cant get episode score > 0.2 so target avg seems miles away. Common Actor Network for both agent but have their own critic 
