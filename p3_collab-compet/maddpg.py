# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'



class MADDPG:
    def __init__(self, number_of_agents, action_size, discount_factor=0.99, tau=0.001):
        super(MADDPG, self).__init__()
        # critic input = obs_full + actions = obs_full (which is concat 2*24 agent_states)+2+2=52
        self.number_of_agents = number_of_agents
        self.action_size = action_size
        self.maddpg_agent = [DDPGAgent(
            in_actor=24,
            hidden_in_actor=256,
            hidden_out_actor=128,
            out_actor=2,
            in_critic_state=24*2,
            hidden_in_critic=256,
            hidden_out_critic=128,
            critic_input_action_size=number_of_agents*action_size) for i in range(number_of_agents)]
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
        actions = [agent.act(obs, noise, grad_zero=True) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
        # print(type(obs), type(obs_full), type(action), type(reward), type(next_obs), type(next_obs_full), type(done))
        # print(len(obs), len(obs_full), len(action), len(reward), len(next_obs), len(next_obs_full), len(done))
        # print(obs[0].shape, obs_full[0].shape, action[0].shape, reward[0].shape, next_obs[0].shape, next_obs_full[0].shape, done[0].shape)
        # print("secondary action", action)

        obs_full = torch.stack(obs_full)
        next_obs_full = torch.stack(next_obs_full)
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1).to(device)
        # print("target_actions", target_actions.shape)
        next_obs_full_t = next_obs_full.t().to(device)
        # print("next_obs_full_t", next_obs_full_t.shape)
        with torch.no_grad():
            # we do not want to update the target model that is the reason behind no grad here
            q_next = agent.target_critic(next_obs_full_t, target_actions)
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        # print("y", y)
        action = torch.cat(action, dim=1).to(device)
        # print("concat action", action)
        obs_full_t = obs_full.t().to(device)
        q = agent.critic(obs_full_t, action)
        # print("obs_full_t", obs_full_t.shape)

        huber_loss = torch.nn.SmoothL1Loss()
        # mse_loss = torch.nn.MSELoss()
        critic_loss = huber_loss(q, y.detach())
        # critic_loss = mse_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # q_action_input = [self.maddpg_agent[i].actor(ob) if i == agent_number\
        #            else self.maddpg_agent[i].actor(ob).detach()
        #            for i, ob in enumerate(obs)]

        if agent_number == 0:
            actions_pred = agent.actor(obs[0])
            action_clone = torch.cat((actions_pred, action[:, 2:]), dim=1)
        else:
            actions_pred = agent.actor(obs[1])
            action_clone = torch.cat((action[:, :2], actions_pred), dim=1)

        # q_action_input = torch.cat(q_action_input, dim=1)
        actor_loss = -agent.critic(obs_full.t(), action_clone).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
        agent.actor_optimizer.step()


    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
    def reset(self):
        for agent in self.maddpg_agent:
            agent.reset()




