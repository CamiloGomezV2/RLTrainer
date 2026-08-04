'''
© Edgar Andrade 2018
Email: edgar.andrade@urosario.edu.co

-----------------------------------------------
Reinforce-learning agents

Includes:
    - MC, a learning rule with Monte Carlo optimization.
    - SARSA, a learning rule 
    - Q_learning, a learning rule
-----------------------------------------------
'''
import numpy as np
from agents.BaseAgent import Agent

dash_line = '-'*20

class MC(Agent) :
    '''
    Implements a learning rule with Monte Carlo optimization.
    '''

    def __init__(self, parameters:dict):
        super().__init__(parameters)
        self.first_visit = self.parameters['first_visit']
        self.N = np.zeros((self.nS, self.nA))
        self.debug = False
   
    def restart(self):
        super().restart()
        self.N = np.zeros((self.nS, self.nA))

    def reset(self):
        super().reset()
        self.N = np.zeros((self.nS, self.nA))

    def update(self, next_state, reward, done):
        '''
        Agent updates its model.
        '''
        if done:
            rewards = [r for r in self.rewards] + [reward]
            T = len(rewards) - 1
            G = 0
            for t in range(T - 1, -1, -1):
                reward = rewards[t+1]
                G  = self.gamma*G + reward
                state = self.states[t]
                if (not self.first_visit) or state not in self.states[:t]:
                    action = self.actions[t]
                    self.N[state, action] += 1
                    prev_Q = self.Q[state, action]
                    self.Q[state, action] += 1/self.N[state, action] * (G - self.Q[state, action])
                    if self.debug:
                        print('')
                        print(dash_line)
                        print(f'Learning log round {t}:')
                        print(f'state:{state}')
                        print(f'action:{action}')
                        print(f'reward:{reward}')
                        print(f'G:{G}')
                        print(f'Previous Q:{prev_Q}')
                        print(f'New Q:{self.Q[state, action]}')
            for s in range(self.nS):
                self.update_policy(s)


class SARSA(Agent) :
    '''
    Implements a SARSA learning rule.
    '''

    def __init__(self, parameters:dict):
        super().__init__(parameters)
        self.alpha = self.parameters['alpha']
        self.debug = False
   
    def update(self, next_state, reward, done):
        '''
        Agent updates its model.
        '''
        # obtain previous state
        state = self.states[-1]
        # obtain previous action
        action = self.actions[-1]
        # Get next_action
        next_action = self.make_decision()
        # Find bootstrap
        estimate = reward + self.gamma * self.Q[next_state, next_action]
        # Obtain delta
        delta = estimate - self.Q[state, action]
        # Update Q value
        prev_Q = self.Q[state, action]
        self.Q[state, action] = prev_Q + self.alpha * delta
        # Update policy
        self.update_policy(state)
        if self.debug:
            print('')
            print(dash_line)
            print(f'Learning log:')
            print(f'state:{state}')
            print(f'action:{action}')
            print(f'reward:{reward}')
            print(f'estimate:{estimate}')
            print(f'Previous Q:{prev_Q}')
            print(f'delta:{delta}')
            print(f'New Q:{self.Q[state, action]}')


class Q_learning(Agent) :
    '''
    Implements a Q-learning rule.
    '''

    def __init__(self, parameters:dict):
        super().__init__(parameters)
        self.alpha = self.parameters['alpha']
        self.debug = False
   
    def update(self, next_state, reward, done):
        '''
        Agent updates its model.
        '''
        # obtain previous state
        state = self.states[-1]
        # obtain previous action
        action = self.actions[-1]
        # Find bootstrap
        maxQ = self.max_Q(next_state) 
        estimate = reward + self.gamma * maxQ
        # Obtain delta
        delta = estimate - self.Q[state, action]
        # Update Q value
        prev_Q = self.Q[state, action]
        self.Q[state, action] = prev_Q + self.alpha * delta # Actualizar en la dirección de delta por una fracción alfa
        # Update policy
        self.update_policy(state)
        if self.debug:
            print('')
            print(dash_line)
            print(f'Learning log:')
            print(f'state:{state}')
            print(f'action:{action}')
            print(f'reward:{reward}')
            print(f'estimate:{estimate}')
            print(f'Previous Q:{prev_Q}')
            print(f'delta:{delta}')
            print(f'New Q:{self.Q[state, action]}') 