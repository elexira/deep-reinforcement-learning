### Model Architecture
See the update rule below
![alt text](./images/update_rule.png "Title")
The important issue here is that the target model is called target in the code.
This model is not updated often 
```buildoutcfg
self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
```
and the old value is called the local model.
This is the model that in vanila version used for identifying the next action to make.
```buildoutcfg
action_values = self.qnetwork_local(state)
np.argmax(action_values.cpu().data.numpy())
```
in the vanilla code the target model and the local model are both updated at the same time,
and after receiving a random batch of experiences (size is batch).
However, they are not update in the same way. The local model is updated with back-propagation.
The target model is  updated softly (weighted average):
```buildoutcfg
θ_target = τ*θ_local + (1 - τ)*θ_target
where τ = 0.001
```
#### Hyper-parameters 
i used the same parameters used in the DQN exercise, and was able to achieve a score of 16. 

### Enhancements 
#### Double DQN
![alt text](./images/double_dqn.png "Title")

### Plot of Rewards