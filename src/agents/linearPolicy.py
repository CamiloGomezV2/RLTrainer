'''
Classes for implementing the Q functions
'''
import json
import numpy as np
from . tiles3 import IHT, tiles
from typing import Dict, List
import torch

class UniformPolicy():
    '''
    Defines a policy that returns 0 for all states
    '''
    def __init__(self, nA):
        self.nA = nA

    def predict(self, state, action):
        return np.zeros(self.nA)
    
    def learn(self, state, action, update, alpha):
        pass

    def save(self, file):
        pass

    def load(self, file):
        pass


class LinearV :
    '''
    Defines a value approximation function
    using the dimensions as features.
    Input:
        - nS (int), number of dimensions
        - nA (int), number of actions
        - alpha (float), the learning rate
    '''
    def __init__(self, parameters:Dict) -> None:
        self.nS = parameters['nS']
        self.alpha = parameters['alpha']
        self.weights = np.random.sample(self.nS)

    def predict(self, state:any) -> List[float]:
        '''
        Returns the value of the state.
        Input:
            - state, the state of the environment as a tuple
        '''
        state_ = np.array(state).reshape(self.nS)
        return np.dot(self.weights, state_)

    def learn(self, state:List[float], delta:float) -> None:
        '''
        Update weights with Gradien Ascent.
        '''
        self.weights += self.alpha * delta * self.predict(state)

    def reset(self) -> None:
        self.weights = np.random.sample(self.nS)


class LinearPolicy :
    '''
    Defines a policy using the dimensions as features.
    Input:
        - nS (int), number of dimensions
        - nA (int), number of actions
        - alpha (float), the learning rate
        - beta (float), the inverse temperature for softmax
        - gamma (float), the discount factor
    '''
    def __init__(self, parameters:Dict) -> None:
        self.nS = parameters['nS']
        self.nA = parameters['nA']
        self.alpha = parameters['alpha']
        self.beta = parameters['beta']
        self.gamma = parameters['gamma']
        self.weights = torch.rand((self.nA, self.nS), 
                                  dtype=torch.float32,
                                  requires_grad=True)
        self.discount = 1

    def predict(self, state:any) -> List[float]:
        '''
        Returns the probabilities.
        Input:
            - state, the state of the environment as a tuple
        '''
        state_ = torch.tensor(state, 
                              dtype=torch.float32,
                              requires_grad=False)
        beta = torch.tensor(self.beta,
                            dtype=torch.float32,
                            requires_grad=False)
        # Find logits of actions
        logits = torch.matmul(self.weights, state_)
        # Return softmax
        return torch.nn.functional.softmax(beta * logits, dim=0)
    
    def learn(self, state:List[float], action:int, delta:float) -> None:
        '''
        Update weights with Gradien Ascent.
        '''
        # Retain grad for weights
        self.weights.retain_grad()
        # Get the predicted probability for the action
        probabilities = self.predict(state)
        # print(f'probabilities:{probabilities}')
        action_prob = probabilities[action]
        # print(f'action_prob:{action_prob}')
        # Determine loss
        loss = torch.log(action_prob)
        # Find the gradients by backward propagation
        loss.backward()
        gradient = self.weights.grad
        # print(f'alpha:{self.alpha} --- discount:{self.discount} --- delta:{delta} --- gradient:{gradient}')
        # print(f'change:{self.alpha * self.discount * delta * gradient}')
        # Update the weights with gradient ascent
        self.weights = self.weights + self.alpha * self.discount * delta * gradient
        # update discount
        self.discount *= self.gamma
        # Manually zero the gradients after updating weights
        self.weights.grad = None

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

    def reset(self) -> None:
        self.discount = 1
        self.weights = torch.rand((self.nA, self.nS), 
                                  dtype=torch.float32,
                                  requires_grad=True)


class TilesPolicy():
    '''
    Defines a tile coding linear approximation.
    Input:
        - nS (int), number of dimensions of the continuous state space
        - numTilings (int), the number of tilings. Should be a power of 2, e.g., 16.
        - numTiles (list), a list with the number of tiles per dimension
        - scaleFactors (list), a list with the normalization factor per dimension
        - maxSize (int), the max number of tiles
        - weights (list), the list of wheights
    '''
    def __init__(self, parameters:Dict):
        self.nA = parameters['nA']
        self.nS = parameters["nS"]
        self.numTilings = parameters["numTilings"]
        self.numTiles = parameters["numTiles"]
        self.scaleFactors = parameters["scaleFactors"]
        self.alpha = parameters["alpha"]
        self.beta = parameters['beta']
        self.gamma = parameters['gamma']
        self.maxSize = parameters["maxSize"]
        self.iht = IHT(self.maxSize)
        self.weights = np.zeros(self.maxSize)
        self.active_tiles = []
        self.discount = 1
            
    def my_tiles(self, state, action) -> List[int]:
        '''
        Determines the tiles that get activated by the state
        '''
        # Normalizes the state
        scaled_s = self.normalize(state)
        # Rescale for use with `tiles` using numTiles
        rescaled_s = [scaled_s[i]*self.numTiles[i] for i in range(self.nS)]
        self.active_tiles = tiles(self.iht, self.numTilings, rescaled_s, [action])
        return self.active_tiles
    
    def predict(self, state:any) -> List[float]:
        '''
        Returns the sum of the weights corresponding to the active tiles
        '''
        get_logit = lambda action: sum([self.weights[tile] for tile in self.my_tiles(state, action)])
        # Find logits of actions
        logits = np.array([get_logit(action) for action in range(self.nA)])
        # Return softmax
        numerator = np.exp(self.beta * logits)
        denominator = np.sum(np.exp(self.beta * logits))
        assert(denominator != 0), f'logits:{logits} --- exp:{np.exp(self.beta * logits)}'
        assert(not np.isnan(denominator)), f'logits:{logits} --- exp:{np.exp(self.beta * logits)}'
        return numerator / denominator

    def learn(self, state, action, G):
        '''
        Updates its weights.
        '''
        probabilities = self.predict(state)
        # Find policy gradient with equation (13.9) from Sutton & Barto (2018)
        X = np.zeros(self.maxSize)
        action_tiles = self.my_tiles(state, action)
        X[action_tiles] = 1
        total_tiles = [t for t in action_tiles]
        for b in range(self.nA):
            tiles_to_add = self.my_tiles(state, b)
            X[tiles_to_add] += probabilities[b] 
            total_tiles += tiles_to_add
        gradient = X[total_tiles]
        # Perform gradient ascent
        # print(f'alfa:{self.alpha}; discount:{self.discount}; G:{G}, gradient:{gradient}')
        self.weights[total_tiles] += self.alpha * self.discount * G * gradient
        # update discount
        self.discount *= self.gamma
        
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
