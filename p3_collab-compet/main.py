# main function that sets up environments
# perform training loop

import envs
from buffer import ReplayBuffer
from maddpg import MADDPG
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
from utilities import transpose_list, transpose_to_tensor

# keep training awake
# from workspace_utils import keep_awake

# for saving gif
import imageio

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def pre_process(entity, batchsize):
    processed_entity = []
    for j in range(3):
        list = []
        for i in range(batchsize):
            b = entity[i][j]
            list.append(b)
        c = torch.Tensor(list)
        processed_entity.append(c)
    return processed_entity


def main():
    seeding()
    # number of parallel agents
    parallel_envs = 4
    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    number_of_episodes = 10000
    episode_length = 100
    batchsize = 1000
    # how many episodes to save policy and gif
    save_interval = 5000
    # what is this ?
    t = 0
    
    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 2
    noise_reduction = 0.9999

    # how many episodes before update
    episode_per_update = 2 * parallel_envs

    log_path = os.getcwd()+"/log"
    model_dir= os.getcwd()+"/model_dir"
    
    os.makedirs(model_dir, exist_ok=True)

    torch.set_num_threads(parallel_envs)
    # this may be a list of all environments
    env = envs.make_parallel_env(parallel_envs)
    
    # keep 5000 episodes worth of replay
    buffer = ReplayBuffer(int(5000*episode_length))
    
    # initialize policy and critic
    # this creates a list of models, each element in the list refers to an agent in the simulation
    # [agent_one_ddpg, agent_two_ddpg, ...]
    # agent_one_ddpg contains the agent actor and critic models,e.g., agent_one_ddpg.actor, agent_one_ddpg.critic
    maddpg = MADDPG()
    logger = SummaryWriter(log_dir=log_path)
    agent0_reward = []
    agent1_reward = []
    agent2_reward = []

    # training loop
    # show progressbar
    import progressbar as pb
    widget = ['episode: ', pb.Counter(),'/',str(number_of_episodes),' ', 
              pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ' ]
    
    timer = pb.ProgressBar(widgets=widget, maxval=number_of_episodes).start()

    # use keep_awake to keep workspace from disconnecting
    # for episode in keep_awake(range(0, number_of_episodes, parallel_envs)):
    # notice we jump forward by number of parallel environments
    for episode in range(0, number_of_episodes, parallel_envs):
        timer.update(episode)

        # i believe there are as many as number of agents times parallel env reward
        reward_this_episode = np.zeros((parallel_envs, 3))
        # obs is the observation state space of all the three agents in the 4 parallel env.
        # for the Physical Dception environment with three agents it is of dimension 4x3x14.
        # obs_full is world state irrespective of the agents and its dimension is 4x14.
        # all_observation = array(number of environments 4, 2 elements)
        # element 0 : is a list that contains 3 arrays. contains the state for each agent, each state is of size 14
        # element 1 : global state from the perspective of the target/green for its environment. contains 14 elements
        all_obs = env.reset()
        # obs : is a list that has 1 element per environment. each element contains a list of 3 array.
        # each array is the state of one agent in that environment.
        # obs_full: is the god eye view of each environment. So it a list, that has 1 element per environment
        # each element contains an array of 14 values which is the global state of that environment
        obs, obs_full = transpose_list(all_obs)

        #for calculating rewards for this particular episode - addition of all time steps

        # save info or not
        save_info = (episode % save_interval < parallel_envs or episode == number_of_episodes-parallel_envs)
        frames = []
        tmax = 0
        
        if save_info:
            frames.append(env.render('rgb_array'))


        
        for episode_t in range(episode_length):
            # we finish the episode before sampling the buffer for trainint
            # t jumps forward in a multiple of environment
            t += parallel_envs
            

            # explore = only explore for a certain number of episodes
            # action input needs to be transposed
            # the transpose_to_tensor(obs) changes the data to each agent point of view
            # since we have 4 environments, there are 4 agent 1, 4 agent 2, and 4 agent 3
            # each agent has a state in each environment, total states across 4 environments for agent 1 is 4x14 tensor
            # transpose_to_tensor(obs) = is a list of 3 elements. each element is for 1 agent
            # pick element 1. this is an array of 4x14 elements of agent observation across 4 environments.
            # maddpg.act has a for loop that take each element of obs and pass it to the agents actor models and
            # to generate an action from each agent actor.
            actions = maddpg.act(transpose_to_tensor(obs), noise=noise)
            noise *= noise_reduction
            # there are 4 actions per agent and 3 agents, total of 12 . Each action has 2 elements force in x, y direct
            # actions_array is a tensor of shape (3 agent, 4 env, 2 action)
            actions_array = torch.stack(actions).detach().numpy()

            # transpose the list of list
            # flip the first two indices
            # input to step requires the first index to correspond to number of parallel agents
            # the shape of actions_for_env is (4 env, 3 agent, 2 action)
            actions_for_env = np.rollaxis(actions_array, 1)
            
            # step forward one frame
            # obs is the observation state space of all the three agents in the 4 parallel env.
            # for the Physical Dception environment with three agents it is of dimension 4x3x14.
            # obs_full is world state irrespective of the agents and its dimension is 4x14.
            # To gain more understanding, please see the code in the multiagent folder.
            next_obs, next_obs_full, rewards, dones, info = env.step(actions_for_env)
            
            # add data to buffer
            transition = (obs, obs_full, actions_for_env, rewards, next_obs, next_obs_full, dones)
            
            buffer.push(transition)
            
            reward_this_episode += rewards

            obs, obs_full = next_obs, next_obs_full
            
            # save gif frame
            if save_info:
                frames.append(env.render('rgb_array'))
                tmax += 1
        
        # update once after every episode_per_update
        if len(buffer) > batchsize and episode % episode_per_update < parallel_envs:
            for a_i in range(3):
                # although samples are drawn randomly, for each sample we have all 3 agents data, and we know which
                # reward and actions belong to which agent
                # samples is a list of 7 elements: obs, obs_full, action, reward, next_obs, next_obs_full, done
                # each element of sample, say samples[0] is a list of 3 elements, one for each agent
                # each agent element contains their corresponding value, for example in case of obs it would be a
                # vector with 14 values
                # so when i ask for 2 samples for examples, i get 2 samples each containing all 3 agents states, rewards
                samples = buffer.sample(batchsize)
                maddpg.update(samples, a_i, logger)
            maddpg.update_targets() #soft update the target network towards the actual networks

        for i in range(parallel_envs):
            agent0_reward.append(reward_this_episode[i, 0])
            agent1_reward.append(reward_this_episode[i, 1])
            agent2_reward.append(reward_this_episode[i, 2])

        if episode % 100 == 0 or episode == number_of_episodes-1:
            avg_rewards = [np.mean(agent0_reward), np.mean(agent1_reward), np.mean(agent2_reward)]
            agent0_reward = []
            agent1_reward = []
            agent2_reward = []
            for a_i, avg_rew in enumerate(avg_rewards):
                logger.add_scalar('agent%i/mean_episode_rewards' % a_i, avg_rew, episode)

        #saving model
        save_dict_list =[]
        if save_info:
            for i in range(3):

                save_dict = {'actor_params': maddpg.maddpg_agent[i].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params': maddpg.maddpg_agent[i].critic.state_dict(),
                             'critic_optim_params': maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)

                torch.save(save_dict_list, 
                           os.path.join(model_dir, 'episode-{}.pt'.format(episode)))
                
            # save gif files
            imageio.mimsave(os.path.join(model_dir, 'episode-{}.gif'.format(episode)), 
                            frames, duration=.04)

    env.close()
    logger.close()
    timer.finish()

if __name__=='__main__':
    main()
