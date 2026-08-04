'''
Classes for implementing the Q functions with neural networks
'''
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from . networks import MLP, CustomCNN, Atari32CNN
from copy import deepcopy
from prettytable import PrettyTable
from tqdm.auto import tqdm

class NNasQ :
    def __init__(self, parameters):
        self.parameters = parameters
        self.NN = parameters["NN"]
        network_parameters = self.NN.model.parameters()
        self.nA = parameters["nA"]
        self.loss_fn = parameters["loss_fn"]
        alpha = parameters["alpha"]
        optimizer_class = parameters["optimizer_class"]
        if optimizer_class == 'Adam':
            self.optimizer = torch.optim.Adam(network_parameters, lr=alpha)
        elif optimizer_class == 'SGD':
            self.optimizer = torch.optim.SGD(
                network_parameters, 
                lr=alpha,
                momentum=0
            )
        else:
            raise Exception(f'Optimizer class {optimizer_class} not implemented!')
        self.backup_NN = deepcopy(self.NN)
        self.losses = []

    def predict(self, state, action):
        with torch.no_grad():
            # Get predicted Q values
            Qs = self.NN(state)
            if len(Qs.shape) > 1:
                Qs = Qs.squeeze()
            # Transform to list
            Qs = Qs.data.tolist()
        return Qs[action]
    
    def values_vector(self, state):
        with torch.no_grad():
            # Get predicted Q values
            Qs = self.NN(state)
            if len(Qs.shape) > 1:
                Qs = Qs.squeeze()
            # Transform to list
            Qs = Qs.data.tolist()
        return Qs
    
    def learn(self, ds_loader:DataLoader):
        '''
        Trains the NN with the given dataset
        '''
        for batch_states, batch_actions, batch_updates in ds_loader:
            # print(f'batch_states:{batch_states}')
            # print(f'batch_actions:{batch_actions}')
            # print(f'batch_updates:{batch_updates}')
            # Clear the gradient
            self.optimizer.zero_grad()
            # Get the batch predicted values
            batch_qvals = self.NN.forward(batch_states)
            # print(f'batch_qvals:{batch_qvals}')
            # Confirm the number of actions
            nA = self.parameters["nA"]
            # Get the indices for the actions
            mask = torch.tensor([i * nA + a for i, a in enumerate(batch_actions)], dtype=torch.long)
            if torch.cuda.is_available():
                mask = mask.to("cuda")
            elif torch.backends.mps.is_available():
                mask = mask.to("mps")
            # Keep only the q values corresponding to the actions
            batch_X = torch.take(batch_qvals, mask)
            # Adapt the update
            batch_Y = batch_updates
            if torch.cuda.is_available():
                batch_Y = batch_Y.to("cuda")
            elif torch.backends.mps.is_available():
                batch_Y = batch_Y.to("mps")
            # Determine loss
            # print(f'X:{batch_X} --- Y:{batch_Y}')
            loss = self.loss_fn(batch_X, batch_Y)
            self.losses.append(loss.item())
            # Find the gradients by backward propagation
            loss.backward()
            # Update the weights with the optimizer
            self.optimizer.step()

    def save(self, file:Path):
        torch.save(self.NN.state_dict(), file)

    def load(self, file:Path):
        self.NN.load_state_dict(torch.load(file))

    def restart(self):
        pass

    def reset(self):
        self.restart()
        # Instantiate original model
        self.NN = deepcopy(self.backup_NN)
        network_parameters = self.NN.model.parameters()
        # Create optimizer
        alpha = self.parameters["alpha"]
        optimizer_class = self.parameters["optimizer_class"]
        if optimizer_class == 'Adam':
            self.optimizer = torch.optim.Adam(network_parameters, lr=alpha)
        self.losses = []

    def summary(self):
        table = PrettyTable(['Modules', 'Parameters'])
        total_params = 0
        for name, parameter in self.NN.model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print(f'Total Trainable Params: {total_params}')
        if next(self.NN.model.parameters()).is_cuda:
            print('Model device: cuda')
        elif next(self.NN.model.parameters()).is_mps:
            print('Model device: mps')
        else:
            print('Model device: cpu')
            

class MLPasQ(NNasQ):
    '''
    Defines a Q function using a Multi-layer Perceptron
    '''
    def __init__(self, parameters):
        self.sizes = parameters["sizes"]
        assert(parameters['nA'] == self.sizes[-1]), f'The last layer in sizes ({self.sizes[-1]}) should be equal to the number of actions ({parameters["nA"]})'
        intermediate_activation_function = parameters["intermediate_activation_function"]
        last_activation_function = parameters["last_activation_function"]
        # Create Multi-layer Perceptron
        NN = MLP(sizes=self.sizes,
                 intermediate_activation_function=intermediate_activation_function, 
                 last_activation_function=last_activation_function)
        parameters["NN"] = NN
        # Define loss function as Mean Square Error
        parameters["loss_fn"] = torch.nn.MSELoss()
        super().__init__(parameters)
       

class CustomCNNasQ(NNasQ):
    '''
    Defines a Q function using a custom Convolutional Network
    '''
    def __init__(self, parameters):
        # Create Convolutional Network
        NN = CustomCNN()
        parameters["NN"] = NN
        # Define loss function as Mean Square Error
        parameters["loss_fn"] = torch.nn.MSELoss()
        super().__init__(parameters)


class Atari32CNNasQ(NNasQ):
    '''
    Defines a Q function using a custom Convolutional Network
    '''
    def __init__(self, parameters):
        # Create Convolutional Network
        NN = Atari32CNN()
        parameters["NN"] = NN
        # Define loss function as Mean Square Error
        parameters["loss_fn"] = torch.nn.MSELoss()
        super().__init__(parameters)
