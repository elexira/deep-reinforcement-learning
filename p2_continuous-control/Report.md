## Project Summary:
### Conda environment setup 
```buildoutcfg
    conda create python=3.6 -n unity
    conda activate unity
    conda install numpy
    pip install unityagents
    pip install mlagents
```
### Project Implementation 
I essentially used the DDPG implementation along with the recommendation from the udacity. 
The key features of my implementation are as follows:
1) using the 20 agent implementation of unity instead of 1 agent. 
this allows me to collect 20 times more experiment in the same amount of time.
2) i unwrap the 20 experiments inside the step function and add them one by one to the replay buffer
3) after each action and collected reward and adding 20 new data points to the replay buffer,
i run the training `n` number of times. One line of code, the for loop, was taken from another udacity student because i
did not quite understood what the instruction means by "update the networks 10 times after every 20 timesteps".
I suggest the authors change the wording to clarify that 20 time steps is actually one round of simulation
in the 20 agent simulation run. 
```
for _ in range(15):
    experiences = self.memory.sample()
```         
4) I tried with initial random action because i felt the actor model will have some initial bias.
the initial bias might prevent the actor from axploring some actions:
```
    def act(self, state, add_noise=True, random_action=False):
        """Returns actions for given state as per current policy."""
        if random_action and len(self.memory) < 2*int(1e4):
            action = np.random.randn(self.number_of_agents, self.action_size)
```
however trying this strategy did not help improve the performance, so i turned it off. 
5) gradient clipping was uses as suggested by Udacity:
```
    # Minimize the loss
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
    self.critic_optimizer.step()
```
6) the noisy action function was adjusted to inject independent noise for each agent:
```
    if add_noise:
        noise = np.array([self.noise.sample() for _ in range(self.number_of_agents)])
        action += noise
```

### Performance
it was very difficult to get the performance improve over time. The biggest problem was that my average score gets stuck and would not improve within the first 20
episodes and won't improve. 

### Concepts and Learnings:
#### What is Replay Buffer 
notice throughout this whole training we keep on accumulating 100,000 experience tuples 
is the limit of how many experiences we collect before stating to throw away the earlier ones
notice that state, action, reward, next_state, done , for a deterministic simulator
is not at all a function of policy or Q. policy, actor, critic may impact how frequently
we perform an action but once the action is performed the next state and reward is derived from the
simulation. For example for a deterministic simulation if we are in state S , take action A, we will
end up in state S' , and this is independent of the models we are training.
so this means the experiences we are collecting better cover a large range of actions and states
so we can learn from them. This is why we play freely and then we learn, and we use epsilon greedy to try
random things so we get to collect reward and better estimate Q(S, A) for a wide combination of S, A
after several iterations, actor starts to reduce the number of random actions and we start acting
optimally according to actor function approximation.


#### Actor Loss function Explained 
we backpropagate the actor local model parameters only, excluding the critic model parameters
in a way to maximixe the Q(s, a) because we want the actor to always spit out argmax Q(s, a)
so we have this critical_local function that can calculate Q(s,a) and i want to train my actor
to output a value to Q(S, a) that will maximize its value for that state S.
while this propagation will impact all parameters of actor for a given state S,
the impact on critic is when actor spits out
an action which is fed into the Q that is in the direction of maximizing Q(s, action_from_actor)
also notice state S and output of the critic model is not part of optimization
when you do backproagate the actor in the optimization. The output of the critic Q(S, A) can
written as a function of the last layer parameteres of actor something like Q(S, A) = 10xWxS ,
in which W is an actor parameter, and S is the visited state and 10xWxS evaluated at current estimated
value of W is the critic output
, e.g., if w_hat = 2 then Q(S, A) = 10x2xS = 20S. Hopefully you can see from this example taking deravatives
with respect to actor parameter W to maximize the Q = 10W is straightforward. 
we could maximize the sum of Q(s, a) over all experiences or mean of Q(s, a) over all experiences

#### Personal Notes from the Lectures
![reinforce](./images/reinforce.png)
![actor_critic](./images/actor_critic.png)