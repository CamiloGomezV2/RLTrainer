'''
Classes for implementing the Q functions
'''
import json
import numpy as np
from . tiles3 import IHT, tiles
from typing import Dict

class UniformQ():
    '''
    Defines a Q function that returns 0 for all state, action pairs
    '''
    def __init__(self, parameters):
        pass

    def predict(self, state, action):
        return 0
    
    def learn(self, state, action, update, alpha):
        pass

    def save(self, file):
        pass

    def load(self, file):
        pass

class DimIdentityQ():
    '''
    Defines a Q function using the dimensions as features.
    Input:
        - nS (int), number of dimensions
        - alpha (float), the learning rate
    '''
    def __init__(self, nS:int, alpha) -> None:
        self.dim = nS + 1
        self.alpha = alpha
        self.weights = np.zeros(self.dim)

    def predict(self, state:any, action:int) -> float:
        '''
        Returns the approximate value.
        Input:
            - state, the state of the environment as a tuple
            - action, the action taken
        '''
        vector = (*state, action)
        return np.dot(self.weights, vector)
    
    def learn(self, state, action, G):
        '''
        Update weights. Gradient is linear and we 
        can use Gradien Descent.
        '''
        current_estimate = self.predict(state, action)
        delta = G - current_estimate
        gradient = np.array((*state, action))
        self.weights += self.alpha * delta * gradient

    def save(self, file:str) -> None:
        json_object = json.dumps(self.weights.tolist(), indent=4)
        # Writing to file
        with open(file, "w") as outfile:
            outfile.write(json_object)
        outfile.close()

    def load(self, file:str) -> None:
        with open(file, 'r') as openfile:
            # Reading from file
            self.weights = json.load(openfile)
        openfile.close()


class TilesQ():
    '''
    Defines a tile coding linear approximation.
    Input:
        - numDims (int), number of dimensions of the continuous state space
        - numTilings (int), the number of tilings. Should be a power of 2, e.g., 16.
        - numTiles (list), a list with the number of tiles per dimension
        - scaleFactors (list), a list with the normalization factor per dimension
        - maxSize (int), the max number of tiles
        - weights (list), the list of wheights
    '''
    def __init__(self, parameters:Dict):
        self.numDims = parameters["numDims"]
        self.numTilings = parameters["numTilings"]
        self.numTiles = parameters["numTiles"]
        self.scaleFactors = parameters["scaleFactors"]
        self.alpha = parameters["alpha"]
        self.maxSize = parameters["maxSize"]
        self.iht = IHT(self.maxSize)
        self.weights = np.zeros(self.maxSize)
        self.active_tiles = []
            
    def my_tiles(self, state, action):
        '''
        Determines the tiles that get activated by the state
        '''
        # Normalizes the state
        scaled_s = self.normalize(state)
        # Rescale for use with `tiles` using numTiles
        rescaled_s = [scaled_s[i]*self.numTiles[i] for i in range(self.numDims)]
        self.active_tiles = tiles(self.iht, self.numTilings, rescaled_s, [action])
        return self.active_tiles
    
    def predict(self, state, action):
        '''
        Returns the sum of the weights corresponding to the active tiles
        '''
        return sum([self.weights[tile] for tile in self.my_tiles(state, action)])

    def learn(self, state, action, update):
        '''
        Updates its weights.
        '''
        estimate = self.predict(state, action)
        error = update - estimate
        # Gradient is 1 only for active tiles and 0 otherwise
        # Thus only updates weights of active tiles
        self.weights[self.active_tiles] += self.alpha * error
        
    def normalize(self, state):
        '''
        Normalizes state. Should perform the following iteration
        scaled_s = []
        for i, scale in enumerate(self.scaleFactors):
            x = scale(state[i], scale["min"], scale["max"])
            scaled_s.append(x)
        I use list comprehension to optimize speed
        '''
        def re_scale(x, min, max):
            return (x - min) / (max - min)

        return [re_scale(state[i], scale["min"], scale["max"]) for i, scale in enumerate(self.scaleFactors)]
    
    def save(self, file:str) -> None:
        json_dump = {'weights':self.weights.tolist(), 'dictionary':{self.iht.dictionary[key]:key for key in self.iht.dictionary.keys()}, 'overfullcount':self.iht.overfullCount}
        json_object = json.dumps(json_dump, indent=4)
        # Writing to file
        with open(file, "w") as outfile:
            outfile.write(json_object)
        outfile.close()

    def load(self, file:str) -> None:
        self.active_tiles = []
        with open(file, 'r') as openfile:
            # Reading from file
            json_dump = json.load(openfile)
        self.weights = json_dump['weights']
        self.iht = IHT(self.maxSize)
        self.iht.dictionary = json_dump['dictionary']
        self.iht.dictionary = {tuple(self.iht.dictionary[key]):int(key) for key in self.iht.dictionary.keys()}
        self.iht.overfullCount = json_dump['overfullcount']
        openfile.close()
