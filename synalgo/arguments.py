#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
import inspect
from typing import Dict, List, Union, Callable

# Libs
import torch as th
from torch import nn

# Custom
from synalgo.utils import TorchParser

##################
# Configurations #
##################

def fate_lr_decay(initial_lr, lr_decay, epochs):
    """ FATE's learning rate decay equation 
    
    Args:
        initial_lr  (float): Initial learning rate specified
        lr_decay    (float): Scaling factor for Learning rate 
        epochs        (int): No. of epochs that have passed
    Returns:
        Scaled learning rate (float)
    """
    lr = initial_lr / (1 + (lr_decay * epochs))
    return lr

torch_parser = TorchParser()
    
###########################################
# Parameter Abstraction class - Arguments #
###########################################

class Arguments:
    """ 
    PySyft, at its heart, is an extension of PyTorch, which already supports a
    plathora of functions for various deep-learning operations. Hence, it would
    be unwise to re-implement what already exists. However, across all functions
    there exists too many arguments for different functions. This class provides
    a means to localise all required parameters for functions that might be used
    during the federated training.
    
    # Model Arguments (reference purposes only)
    input_size, output_size, is_condensed

    # Optimizer Arguments (only for selected optimizer(s))
    torch.optim.SGD(params, lr=<required parameter>, momentum=0, 
                     dampening=0, weight_decay=0, nesterov=False)
                     
    # Criterion Arguments
    torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    SurrogateCriterion(mu, l1_lambda, l2_lambda)

    # LR Decay Arguments (selected schedulers only)
    torch.optim.lr_scheduler.LambdaLR   (optimizer, lr_lambda, last_epoch=-1)
                            .CyclicLR   (optimizer, base_lr, max_lr, 
                                         step_size_up=2000, step_size_down=None,
                                         mode='triangular', gamma=1.0,
                                         scale_fn=None, scale_mode='cycle', 
                                         cycle_momentum=True, base_momentum=0.8, 
                                         max_momentum=0.9, last_epoch=-1)

    # Early Stopping Arguments
    EarlyStopping (patience, delta)

    # Arguments for functions are retrieved via `func.__code__.co_varnames`    
    """
    def __init__(
        self, 
        algorithm: str = "FedProx", 
        batch_size: int = None, 
        rounds: int = 10, 
        epochs: int = 100,
        lr: float = 0.001, 
        weight_decay: float = 0.0,
        lr_decay: float = 0.1, 
        mu: float = 0.1, 
        l1_lambda: float = 0.0, 
        l2_lambda: float = 0.0,
        optimizer: str = "SGD", 
        criterion: str = "BCELoss", 
        lr_scheduler: str = "CyclicLR", 
        delta: float = 0.0,
        patience: int = 10,
        seed: int = 42,
        is_condensed: bool = True,
        is_snn: bool = False, 
        precision_fractional: int = 5,
        **kwargs
    ):

        # General Parameters
        self.algorithm = algorithm
        self.batch_size = batch_size     # Default: None (i.e. bulk analysis)
        self.rounds = rounds
        self.epochs = epochs
        self.seed = seed
        self.is_condensed = is_condensed
        self.is_snn = is_snn
        self.precision_fractional = precision_fractional

        # Optimizer Parameters
        self.lr = lr
        self.weight_decay = 0 if l2_lambda else weight_decay # for consistency
        self.optimizer = torch_parser.parse_optimizer(optimizer)
        
        # Criterion Parameters
        self.mu = mu
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.criterion = torch_parser.parse_criterion(criterion)

        # LR Decay Parameters
        self.lr_decay = lr_decay
        self.lr_lambda = lambda epochs: fate_lr_decay(
            initial_lr=self.lr, 
            lr_decay=self.lr_decay, 
            epochs=epochs
        )
        self.lr_scheduler = torch_parser.parse_scheduler(lr_scheduler)
        
        # Early Stopping parameters
        self.patience = patience
        self.delta = delta

        # Custom Parameters
        for arg_key, arg_value in kwargs.items():
            setattr(self, arg_key, arg_value)

    ###########
    # Getters #
    ###########

    @property
    def model_params(self):
        return {
            "algorithm": self.algorithm,
            "batch_size": self.batch_size,
            "rounds": self.rounds,
            "epochs": self.epochs,
            "precision_fractional": self.precision_fractional,
            "seed": self.seed
        }


    @property
    def optimizer_params(self):
        return self.__retrieve_args(self.optimizer)
    
    
    @property
    def criterion_params(self):
        params = {
            'mu': self.mu,
            'l1_lambda': self.l1_lambda,
            'l2_lambda': self.l2_lambda
        }
        custom_params = self.__retrieve_args(self.criterion)
        params.update(custom_params)
        return params
        

    @property
    def lr_decay_params(self):
        params = self.__retrieve_args(self.lr_scheduler)

         # Optimizer is dynamically loaded -> remove ambiguity
        params.pop('optimizer')

        return params


    @property
    def early_stopping_params(self):
        return {
            'patience': self.patience,
            'delta': self.delta
        }

    ###########
    # Helpers #
    ###########

    def __retrieve_args(self, callable: Callable) -> List[str]:
        """ Retrieves all argument keys accepted by a specified callable object
            from a pool of miscellaneous potential arguments

        Args:
            callable (callable): Callable object to be analysed
        Returns:
            Argument keys (list(str))
        """
        input_params = list(inspect.signature(callable).parameters)

        arguments = {}
        for param in input_params:
            param_value = getattr(self, param, None)
            if param_value:
                arguments[param] = param_value

        return arguments
        
#########
# Tests #
#########

if __name__ == "__main__":
    args = Arguments()
    print(args.optimizer_params)
    print(args.lr_decay_params)
    print(args.early_stopping_params)
