'''
Classes for implementing policy gradient methods. 
We assume a discrete action space.
'''
import numpy as np
from . BaseAgent import PolicyAgent

dash_line = '-'*20


class REINFORCE(PolicyAgent) :
    '''
    Implements a REINFORCE rule for policy learning.
    '''
    def __init__(self, parameters:dict):
        super().__init__(parameters)

    def update(self, next_state, reward, done):
        '''
        Agent updates according to the REINFORCE rule.
        '''
        if done:
            rewards = [r for r in self.rewards] + [reward]
            T = len(rewards)
            for t in range(T - 1):
                # Obtain total discounted reward
                G  = np.sum([np.power(self.gamma, k - t - 1) * rewards[k] for k in range(t + 1, T)])
                # Get state_t and action_t from records
                state, action = self.states[t], self.actions[t]
                previous_p = self.policy.predict(state) 
                # Update policy
                self.policy.learn(state, action, G) 
                if self.debug:
                    print('')
                    print(dash_line)
                    print(f'Learning log round {t}:')
                    print(f'state:{state}')
                    print(f'action:{action}')
                    print(f'reward:{reward}')
                    print(f'G:{G}')
                    print(f'Previous probs:{previous_p}')
                    print(f'New probs:{self.policy.predict(state)}')
        