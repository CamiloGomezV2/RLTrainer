from . BaseAgent import PolicyAgent
from copy import deepcopy
import numpy as np


class ActorCritic(PolicyAgent) :
    '''
    Action Critic agent.
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
        self.V = self.parameters['V']
        assert(hasattr(self.V, 'predict')), 'V must be an object with a predict() method'
        assert(hasattr(self.V, 'learn')), 'V must be an object with a learn() method'
        assert(hasattr(self.V, 'save')), 'V must be an object with a save() method'
        assert(hasattr(self.V, 'load')), 'V must be an object with a load() method'
        self.backup_V = deepcopy(self.V)
        self.backup_policy = deepcopy(self.policy)
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [np.nan]
        self.seed = None
        self.debug = False

    def update(self, next_state, reward, done):
        if done:
            # next_state is terminal, no need to bootstrap
            G = reward
        else:
            # Bootstrap 
            G = reward + self.gamma * self.V.predict(next_state)
        # Determine delta
        state = self.states[-1]
        action = self.actions[-1]
        delta = G - self.V.predict(state)
        # Update policy
        self.policy.learn(state, action, delta)
        # Update value estimate
        self.V.learn(state, delta)