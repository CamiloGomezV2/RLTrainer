'''
Implements:
    - the base tabular agent, Agent
    - the base agent for Q approximation, AgentCS
    - the base agent for Q approximations based on Neural Networks, AgentNN
    - the base agent for policy learning, PolicyAgent
'''

import json
import numpy as np
from typing import Dict
from pathlib import Path
from copy import deepcopy
from typing import Optional
from termcolor import colored



class Agent :
    '''
    Defines the basic methods for tabular RL agents.
    '''

    def __init__(self, parameters:Dict[any, any]):
        self.parameters = parameters
        self.nS = self.parameters['nS']
        self.nA = self.parameters['nA']
        self.gamma = self.parameters['gamma']
        self.epsilon = self.parameters['epsilon']
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [np.nan]
        self.policy = np.ones((self.nS, self.nA)) * 1/self.nA
        self.Q = np.zeros((self.nS, self.nA))
        self.seed = None

    def make_decision(self, state:Optional[any]=None):
        '''
        Agent makes a decision according to its policy.
        '''
        if self.seed is not None:
            np.random.seed(self.seed)
        if state is None:
            state = self.states[-1]
        weights = [self.policy[state, action] for action in range(self.nA)]
        action = np.random.choice(range(self.nA), p=weights)
        return action

    def restart(self):
        '''
        Restarts the agent for a new trial.
        '''
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [np.nan]

    def reset(self):
        '''
        Resets the agent for a new simulation.
        '''
        self.restart()
        self.policy = np.ones((self.nS, self.nA)) * 1/self.nA
        self.Q = np.zeros((self.nS, self.nA))

    def max_Q(self, s):
        '''
        Determines the max Q value in state s
        '''
        return max([self.Q[s, a] for a in range(self.nA)])

    def argmaxQ(self, s):
        '''
        Determines the action with max Q value in state s
        '''
        maxQ = self.max_Q(s)
        opt_acts = [a for a in range(self.nA) if self.Q[s, a] == maxQ]
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.choice(opt_acts)

    def update_policy(self, s):
        opt_act = self.argmaxQ(s)
        prob_epsilon = lambda action: 1 - self.epsilon if action == opt_act else self.epsilon/(self.nA-1)
        self.policy[s] = [prob_epsilon(a) for a in range(self.nA)]

    def update(self, next_state, reward, done):
        '''
        Agent updates its model.
        TO BE DEFINED BY SUBCLASS
        '''
        pass

    def save(self, file:Path) -> None:
        # Serializing json
        dictionary = {'policy':self.policy.tolist(),
                      'Q':self.Q.tolist()}
        json_object = json.dumps(dictionary, indent=4)
        # Writing to file
        with open(file, "w") as outfile:
            outfile.write(json_object)
        outfile.close()

    def load(self, file:Path):
        # Opening JSON file
        with open(file, 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
        self.reset()
        self.policy = np.array(json_object['policy'])
        self.Q = np.array(json_object['Q'])
        openfile.close()



class AgentCS() :
    '''
    Super class of agents with Q approximation.
    '''

    def __init__(self, parameters:dict):
        self.parameters = parameters
        self.nS = self.parameters['nS']
        self.nA = self.parameters['nA']
        self.gamma = self.parameters['gamma']
        self.epsilon = self.parameters['epsilon']
        self.alpha = self.parameters['alpha']
        self.Q = self.parameters['Q']
        assert(hasattr(self.Q, 'predict')), 'Q must be an object with a predict() method'
        assert(hasattr(self.Q, 'learn')), 'Q must be an object with a learn() method'
        assert(hasattr(self.Q, 'save')), 'Q must be an object with a save() method'
        assert(hasattr(self.Q, 'load')), 'Q must be an object with a load() method'
        self.backup_Q = deepcopy(self.Q)
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [np.nan]
        self.seed = None
        self.debug = False

    def make_decision(self, state:Optional[any]=None):
        '''
        Agent makes an epsilon greedy accion based on Q values.
        '''
        if np.random.uniform(0,1) < self.epsilon:
            return np.random.choice(range(self.nA))
        else:
            if state is None:
                state = self.states[-1]
            return self.argmaxQ(state)        

    def update(self, next_state, reward, done):
        '''
        Agent updates its Q function according to a model.
            TO BE OVERWRITTEN BY SUBCLASS  
        '''
        pass

    def max_Q(self, s):
        '''
        Determines the max Q value in state s
        '''
        return max([self.Q.predict(s, a) for a in range(self.nA)])

    def argmaxQ(self, s):
        '''
        Determines the action with max Q value in state s
        '''
        maxQ = self.max_Q(s)
        opt_acts = [a for a in range(self.nA) if self.Q.predict(s, a) == maxQ]
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.choice(opt_acts)

    def restart(self):
        '''
        Restarts the agent for a new trial.
        '''
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [np.nan]

    def reset(self):
        '''
        Resets the agent for a new simulation.
        '''
        self.restart()
        # Resets the Q function
        self.Q = deepcopy(self.backup_Q)
    
    def save(self, file:Path) -> None:
        self.Q.save(file=file)

    def load(self, file:Path) -> None:
        self.Q.load(file=file)



class PolicyAgent() :
    '''
    Super class of agents with policy learning.
    '''

    def __init__(self, parameters:dict):
        self.parameters = parameters
        self.nS = self.parameters['nS']
        self.nA = self.parameters['nA']
        self.gamma = self.parameters['gamma']
        self.policy = self.parameters['policy']
        assert(hasattr(self.policy, 'predict')), 'Policy must be an object with a predict() method'
        assert(hasattr(self.policy, 'learn')), 'Policy must be an object with a learn() method'
        assert(hasattr(self.policy, 'save')), 'Policy must be an object with a save() method'
        assert(hasattr(self.policy, 'load')), 'Policy must be an object with a load() method'
        self.backup_policy = deepcopy(self.policy)
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [np.nan]
        self.seed = None
        self.debug = False

    def make_decision(self, state=None):
        '''
        Agent makes an epsilon greedy accion based on Q values.
        '''
        if state is None:
            state = self.states[-1]
        weights = self.policy.predict(state)
        return np.random.choice(range(self.nA), p=weights)
   
    def update(self, next_state, reward, done):
        '''
        Agent updates its Q function according to a model.
            TO BE OVERWRITTEN BY SUBCLASS  
        '''
        pass

    def restart(self):
        '''
        Restarts the agent for a new trial.
        '''
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [np.nan]

    def reset(self):
        '''
        Resets the agent for a new simulation.
        '''
        self.restart()
        # Resets the Q function
        self.policy = deepcopy(self.backup_policy)

    def save(self, file:str) -> None:
        self.policy.save(file=file)

    def load(self, file:str) -> None:
        self.policy.load(file=file)

