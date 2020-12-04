# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'



class MADDPG:
    def __init__(self, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()
        # each agent has it's own critic and actor network'
        # each actor gets its agents state
        # but each agent critic seems to be getting the same input
        # critic input = obs_full + actions = 14+2+2+2=20
        self.maddpg_agent = [DDPGAgent(14, 16, 8, 2, 20, 32, 16), 
                             DDPGAgent(14, 16, 8, 2, 20, 32, 16), 
                             DDPGAgent(14, 16, 8, 2, 20, 32, 16)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

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
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
        # this is global observations at time t
        # there is 1 global for each sample
        # this line converts the list into a tensor. obs_full.shape = 14x5 = [global_state x sample_size]
        obs_full = torch.stack(obs_full)
        # this is global observations at time t+1
        # this line converts the list into a tensor. obs_full.shape = 14x5 = [global_state x sample_size]
        next_obs_full = torch.stack(next_obs_full)
        # extracts the particular agent model instance from the list of agent model instances
        agent = self.maddpg_agent[agent_number]
        # we only activate the specific agent model updating
        agent.critic_optimizer.zero_grad()
        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        # the actor only uses its own agent observation to make a decision with a for loop
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)
        # finally obs_full and next_obs_full converted to [sample_size 5 x global state size 14]
        # target_critic_input.shape = [sample size , 2 action agent1 + 2 act agent2 + 2 action agent3 + 14 global state space size]
        # this is ready for Q(s, a) where s + a = 20 dimenional
        target_critic_input = torch.cat((next_obs_full.t(), target_actions), dim=1).to(device)
        with torch.no_grad():
            # we do not want to update the taregt model that is the reason behind no grad here
            q_next = agent.target_critic(target_critic_input)
        
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        action = torch.cat(action, dim=1)
        critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        # notice y.detach() will prevent y from contributing into the gradient of loss. essentially y becomes constant
        # that prevents parameters of the y (target) model to be updated.
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        # the following is a list of actions recommended by local actor of each agent actor at time t (not t+1)
        # only one agent parameters is activated for back propagation, the others are only used for forward pass
        # to compute their corresponding agents action. But the actions are constants not tensors.
        q_input = [self.maddpg_agent[i].actor(ob) if i == agent_number\
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs)]
        # turns a list into a tensor q_input.shape = [sample_size][3 agents x 2 actions dimension = 6 dimension]
        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # unlike the lecture videos the observation passed to the critic is not agents observation but one single
        # global observation
        # many of the obs are redundant, and obs[1] contains all useful information already
        # turns a list into a tensor q_input.shape = [sample_size][3 agents x 2 actions dimension = 6 dimension + 14 global = 20]
        q_input2 = torch.cat((obs_full.t(), q_input), dim=1)
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        # actor_optimizer only contains the actor parameters so the critic model parameters are not updated.
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            




