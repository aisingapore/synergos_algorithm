#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import importlib
import os
from typing import Any, Dict, List, Union, Tuple, Callable

# Libs


# Custom


##################
# Configurations #
##################


############################################
# Base Configuration Parser Class - Parser #
############################################

class Parser:
    """ 
    Base class that facilitates the loading of modules at runtime given
    their string names
    """

    def parse_operation(self, module_str: str, operation_str: str):
        """ Detects layer type of a specified layer from configuration

        Args:
            module_str    (str): String module to search from
            operation_str (str): String operation to translate
        Returns:
            Module operation
        """
        module = importlib.import_module(module_str)
        operation = getattr(module, operation_str)
        return operation



############################################
# Configuration Parser Class - TorchParser #
############################################

class TorchParser(Parser):
    """ 
    Dynamically translates string names to PyTorch classes

    Attributes:
        MODULE_OF_LAYERS      (str): Import string for layer modules
        MODULE_OF_ACTIVATIONS (str): Import string for activation modules
        MODULE_OF_OPTIMIZERS  (str): Import string for optimizer modules
        MODULE_OF_CRITERIONS  (str): Import string for criterion modules
        MODULE_OF_SCHEDULERS  (str): Import string for scheduler modules
    """
    
    def __init__(self):
        super().__init__()
        self.MODULE_OF_LAYERS = "torch.nn"
        self.MODULE_OF_ACTIVATIONS = "torch.nn.functional"
        self.MODULE_OF_OPTIMIZERS = "torch.optim"
        self.MODULE_OF_CRITERIONS = "torch.nn"
        self.MODULE_OF_SCHEDULERS = "torch.optim.lr_scheduler"


    def parse_layer(self, layer_str: str) -> Callable:
        """ Detects layer type of a specified layer from configuration

        Args:
            layer_str (str): Layer type to initialise
        Returns:
            Layer definition (Callable)
        """
        return self.parse_operation(self.MODULE_OF_LAYERS, layer_str)


    def parse_activation(self, activation_str: str) -> Callable:
        """ Detects activation function specified from configuration

        Args:
            activation_type (str): Activation function to initialise
        Returns:
            Activation definition (Callable)
        """
        if not activation_str:
            return lambda x: x

        return self.parse_operation(self.MODULE_OF_ACTIVATIONS, activation_str)
    

    def parse_optimizer(self, optim_str: str) -> Callable:
        """ Detects optimizer specified from configuration

        Args:
            optim_str (str): Optimizer to initialise
        Returns:
            Optimizer definition (Callable)
        """
        return self.parse_operation(self.MODULE_OF_OPTIMIZERS, optim_str)


    def parse_criterion(self, criterion_str: str) -> Callable:
        """ Detects criterion specified from configuration

        Args:
            criterion_str (str): Criterion to initialise
        Returns:
            Criterion definition (Callable)
        """
        return self.parse_operation(self.MODULE_OF_CRITERIONS, criterion_str)


    def parse_scheduler(self, scheduler_str: str) -> Callable:
        """ Detects learning rate schedulers specified from configuration

        Args:
            scheduler_str (str): Learning rate scheduler to initialise
        Returns:
            Scheduler definition (Callable)
        """
        if not scheduler_str:
            return self.parse_operation(self.MODULE_OF_SCHEDULERS, "LambdaLR")

        return self.parse_operation(self.MODULE_OF_SCHEDULERS, scheduler_str)



###########################################
# Configuration Parser Class - TuneParser #
###########################################

class TuneParser(Parser):
    """ 
    Dynamically translates string names to Tune API callables

    Attributes:
        MODULE_OF_HYPERPARAM_TYPES (str): Import string for hyperparam types
    """
    
    def __init__(self):
        super().__init__()
        self.MODULE_OF_HYPERPARAM_TYPES = "ray.tune"
        self.MODULE_OF_HYPERPARAM_SCHEDULERS = "ray.tune.schedulers"
        self.MODULE_OF_HYPERPARAM_SEARCHERS = "ray.tune.suggest"


    def parse_type(self, type_str: str) -> Callable:
        """ Detects hyperparameter type of a declared hyperparameter from
            configuration

        Args:
            type_str (str): Layer type to initialise
        Returns:
            Type definition (Callable)
        """
        return self.parse_operation(self.MODULE_OF_HYPERPARAM_TYPES, type_str)


    def parse_scheduler(self, scheduler_str: str) -> Callable:
        """ Detects hyperparameter scheduler from configuration. This variant
            is important as `ray.tune.create_scheduler` requires kwargs to be
            specified to return a fully instantiated scheduler, whereas this
            way the scheduler parameter signature can be retrieved.

        Args:
            scheduler_str (str): Scheduler type to initialise
        Returns:
            Scheduler definition (Callable)
        """
        return self.parse_operation(
            self.MODULE_OF_HYPERPARAM_SCHEDULERS, 
            scheduler_str
        )


    def parse_searcher(self, searcher_str: str) -> Callable:
        """ Detects hyperparameter searcher from configuration. This variant
            is important as `ray.tune.create_searcher` requires kwargs to be
            specified to return a fully instantiated scheduler, whereas this
            way the scheduler parameter signature can be retrieved.

        Args:
            searcher_str (str): Searcher type to initialise
        Returns:
            Searcher definition (Callable)
        """
        SEARCHER_MAPPINGS = {
            'BasicVariantGenerator': "basic_variant",
            'AxSearch': "ax",
            'BayesOptSearch': "bayesopt",
            'TuneBOHB': "bohb",
            'DragonflySearch': "dragonfly",
            'HEBOSearch': "hebo",
            'HyperOptSearch': "hyperopt",
            'NevergradSearch': "nevergrad",
            'OptunaSearch': "optuna",
            'SigOptSearch': "sigopt",
            'SkOptSearch': "skopt",
            'ZOOptSearch': "zoopt"
        }
        partial_import_str = SEARCHER_MAPPINGS[searcher_str]
        return self.parse_operation(
            '.'.join([self.MODULE_OF_HYPERPARAM_SEARCHERS, partial_import_str]),
            searcher_str
        )
