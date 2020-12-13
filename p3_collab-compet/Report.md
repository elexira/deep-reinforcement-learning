# Introduction
This was the final and most challenging project in this course. Overall, after completing the assignments i feel it is very tricky to get the DRL to converge. I see a variety of limitations
such as the need for simulated environments to train the agents. For use-cases in which failure is 
very expensive such as self-driving cars, we do not have the luxuary of simulations/failures. I also feel the convergence is very tricky. Even if we get the optimization to converge in one scenario, if the simulation is changed slightly, there is no garantee that the optimization will be stable. 
I struggled quite a bit in getting the optimizations to work for this class where the action and state space was very small. I cannot imagine the algorithms will work in real world situations where we have myriad of actions and a huge and unpreditable state space. At this moment, i feel the DRL is limited to computer games. 
    
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
    # passing number of agents
    maddpg = MADDPG(number_of_agents, action_size)
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




# =====================================
        # each agent has it's own critic and actor network'
        # each actor gets its agents state
        # but each agent critic seems to be getting the same input
        # critic input = obs_full + actions = obs_full (which is concat 2*24 agent_states)+2+2=52
        # Agent input variables : in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic
        # each tennis player (agent) has 4 models : actor_local, actor_target, critic_local, critic_target

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[sample_size][agent_number] to
        # obs[agent_number][sample_size]
        # obs = [agent_number]([sample_size][14 state elements])
        # obs for 3 agents and sample size 5: a list with 3 elements. each element is (5x14)
        # osb_full = [14][sample_size] ==> later is converted into a tensor of shape (14x5)
        # obs_full for 3 agent sample size 5: a list with 14 elements, each elements is a (1x5)
        # action = [agent_number][sample_size][action_size]
        # reward = [agent_number][sample_size] *reward itself is a single value you do not need any additional dimension
        # next_obs = for 3 agents and sample size 5: a list with 3 elements. each element is (5x14)
        # next_obs_full = same as obs_full
        # done = [agent_number][sample_size] *done itself is a single value you do not need any additional dimension
        # print("code environment obs", obs)
        # print("code environment obs_full", obs_full)
        # print("code environment next obs", next_obs)
        # print("code environment next_obs_full", next_obs_full)
        # print("code environment reward", reward)
        # print("code environment done", done)
        # this is global observations at time t
        # there is 1 global for each sample
        # this line converts the list into a tensor. obs_full.shape = 14x5 = [global_state x sample_size]
        # print("code environment obs_full", obs_full.shape, obs_full)
        # this is global observations at time t+1
        # this line converts the list into a tensor. obs_full.shape = 14x5 = [global_state x sample_size]
        # print("code environment next_obs_full", next_obs_full.shape, next_obs_full)
        # extracts the particular agent model instance from the list of agent model instances
        # we only activate the specific agent model updating
        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        # the actor only uses its own agent observation to make a decision with a for loop
        # print("critic_action_input_shape", target_actions.shape)
        # finally obs_full and snext_obs_full converted to [sample_size 5 x global state size 14]
        # target_critic_input.shape = [sample size , 2 action agent1 + 2 act agent2 + 2 action agent3 + 14 global state space size]
        # this is ready for Q(s, a) where s + a = 20 dimenional
        # predicted reward by target model
        # critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        # local critic model estimation of value action for time t using actual observed state and action
        mse_loss = torch.nn.MSELoss()
        # notice y.detach() will prevent y from contributing into the gradient of loss. essentially y becomes constant
        # that prevents parameters of the y (target) model to be updated.
        # critic_loss = huber_loss(q, y.detach())
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)

        #update actor network using policy gradient
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        # the following is a list of actions recommended by local actor of each agent actor at time t (not t+1)
        # only one agent parameters is activated for back propagation, the others are only used for forward pass
        # to compute their corresponding agents action. But the actions are constants not tensors.
        # turns a list into a tensor q_input.shape = [sample_size][3 agents x 2 actions dimension = 6 dimension]
        # combine all the actions and observations for input to critic
        # unlike the lecture videos the observation passed to the critic is not agents observation but one single
        # global observation
        # many of the obs are redundant, and obs[1] contains all useful information already
        # turns a list into a tensor q_input.shape = [sample_size][3 agents x 2 actions dimension = 6 dimension + 14 global = 20]
        # q_input2 = torch.cat((obs_full.t(), q_action_input), dim=1)
        # notice we have the critic model for 1 agent. but it needs all other agents models actions. instead of using observed action
        # from the random sample
        # here we are using the predicted action from the actor models. we need this because we want action as a function
        # of actor parameters so we can backpropagate and optimize it.
        # get the policy gradient
        # actor_optimizer only contains the actor parameters so the critic model parameters are not updated.
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)

            
#============================================================================
            




