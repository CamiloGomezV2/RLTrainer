import torch
from typing import List, Union
from copy import deepcopy

'''NN arquitectures'''


class MLP(torch.nn.Module):
    '''
    A Multi-layer Perceptron
    '''
    def __init__(self, 
                 sizes:List[int], 
                 intermediate_activation_function:any,
                 last_activation_function:Union[None, any]):
        """
        Args:
            sizes (list): list with the sizes of the layers. 
                          The convention is sizes[0] is the size of the input layer
                          and sizes[-1] is the size of the output layer.
            last_activation_function (an activation function)
        """
        super().__init__()
        assert(len(sizes) > 1)
        self.sizes = sizes
        self.intermediate_activation_function = intermediate_activation_function
        self.last_activation_function = last_activation_function
        # -------------------------------------
        # Defining the layers
        # -------------------------------------
        self.model = torch.nn.Sequential()
        for i in range(len(sizes) - 1):
            n_from = sizes[i]
            n_to = sizes[i+1]
            self.model.append(torch.nn.Linear(n_from, n_to))
            if i < len(sizes) - 2:
                self.model.append(self.intermediate_activation_function)
                # self.model.append(torch.nn.LayerNorm(n_to))
        if self.last_activation_function is not None:
            self.model.append(self.last_activation_function)
        if torch.cuda.is_available():
            self.model.to("cuda")
        elif torch.backends.mps.is_available():
            self.model.to("mps")

    def forward(self, x_in):
        """The forward pass of the network        
        Args:
            x_in (torch.Tensor): an input data tensor. 
        Returns:
            the resulting tensor.
        """
        # Run the input through layers 
        if torch.cuda.is_available():
            x_in = x_in.to("cuda")
        elif torch.backends.mps.is_available():
            x_in = x_in.to("mps")
        return self.model(x_in)
    

class CustomCNN(torch.nn.Module):
    ''' 
    A custom convolutional network to solve PitLaberynth.
    The input image is an array of shape (4, 4, 3)
    '''    
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential()
        #  First convolutional layer
        self.model.append(torch.nn.Conv2d(1,4, kernel_size=2,stride=1, padding=1))
        self.model.append(torch.nn.ReLU())
        self.model.append(torch.nn.MaxPool2d(stride=2, kernel_size=2))
        # Second convolutional layer
        self.model.append(torch.nn.Conv2d(4,8, kernel_size=2,stride=1, padding=1))
        self.model.append(torch.nn.ReLU())
        self.model.append(torch.nn.MaxPool2d(stride=1, kernel_size=2))
        # Linear layers
        self.model.append(torch.nn.Linear(32,64))
        self.model.append(torch.nn.Sigmoid())
        self.model.append(torch.nn.Linear(64,4))

    def forward(self, x_in):
        """The forward pass of the network        
        Args:
            x_in (torch.Tensor): an input data tensor. 
        Returns:
            the resulting tensor.
        """
        # Run the input through layers 
        X = x_in
        for i, layer in enumerate(self.model):
            if i == 6:
                # Flatten before linear layers
                if len(X.shape) == 3:
                    X = torch.flatten(X, start_dim=0)
                elif len(X.shape) == 4:
                    X = torch.flatten(X, start_dim=1)
                else:
                    raise Exception('Whaaaat!')
            X = layer(X)
        return X


class Atari32CNN(torch.nn.Module):
    '''A CNN adapted for a 32x32 1 channel array''' 
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential()
        #  First convolutional layer
        self.model.append(torch.nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1))
        self.model.append(torch.nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1))
        self.model.append(torch.nn.ReLU())
        self.model.append(torch.nn.MaxPool2d(stride=2, kernel_size=2))
        # Create second convolutional layer
        self.model.append(torch.nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1))
        self.model.append(torch.nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1))
        self.model.append(torch.nn.ReLU())
        self.model.append(torch.nn.MaxPool2d(stride=2, kernel_size=2))
        # Create first linear layer
        self.model.append(torch.nn.Linear(1280, 254))
        self.model.append(torch.nn.Sigmoid())
        # Create second linear layer
        self.model.append(torch.nn.Linear(254, 5))

    def forward(self, x_in):
        """The forward pass of the network        
        Args:
            x_in (torch.Tensor): an input data tensor. 
        Returns:
            the resulting tensor.
        """
        # Run the input through layers 
        X = x_in
        for i, layer in enumerate(self.model):
            if i == 8:
                # Flatten before linear layers
                if len(X.shape) == 3:
                    X = torch.flatten(X, start_dim=0)
                elif len(X.shape) == 4:
                    X = torch.flatten(X, start_dim=1)
                else:
                    raise Exception('Whaaaat!')
            X = layer(X)
        return X