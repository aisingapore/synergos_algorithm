#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
from collections import OrderedDict
from typing import Tuple

# Libs
import syft as sy
import torch as th
from torch import nn

# Custom
from synalgo.utils import TorchParser

##################
# Configurations #
##################

torch_parser = TorchParser()

###################################
# Model Abstraction Class - Model #
###################################

class Model(nn.Module):
    """
    The Model class serves to automate the building of structured deep neural
    nets, given specific layer configurations. Being a parent class of sy.Plan,
    this makes it more efficient to deploy in terms of communication costs.

    Args:
        owner (VirtualWorker/WebsocketClientWorker): Handler of this model
        structure (OrderedDict): Configurations used to build the achitecture of the NN
        is_condensed (bool): Toggles Binary or Multiclass prediction

    Attributes:
        is_condensed  (bool): Toggles Binary or Multiclass prediction
        layers (OrderedDict): Maps specific layers to their respective activations
        + <Specific layer configuration dynamically defined>
    """
    def __init__(self, structure):
        super().__init__()
        self.__SPECIAL_CASES = ['RNNBase', 'RNN', 'RNNCell', 
                                'LSTM', 'LSTMCell',
                                'GRU', 'GRUCell']
        
        ###########################
        # Implementation Footnote #
        ###########################

        # [Causes]
        # Any learning rate scheduler with "plateau" (eg. "ReduceLROnPlateau") 
        # requires its model to have the attribute 'self.metric'

        # [Problems]
        # Without this parameter, the following error will be raised:
        # "TypeError: step() missing 1 required positional argument: 'metrics'"
        
        # [Solution]
        # Specify it by default. It will be available for those layers/functions
        # who need it, and ignored by those who do not.

        self.metric = 0  # used for learning rate policy 'plateau'

        self.layers = OrderedDict()

        for layer, params in enumerate(structure):

            # Detect if input layer
            is_input_layer = params['is_input']

            # Detect layer type
            layer_type = params['l_type']

            # Construct layer name (eg. nnl_0_linear)
            layer_name = self.__construct_layer_name(layer, layer_type)

            # Extract layer structure and initialise layer
            layer_structure = params['structure']
            setattr(
                self, 
                layer_name,
                torch_parser.parse_layer(layer_type)(**layer_structure)
            )

            # Detect activation function & store it for use in .forward()
            # Note: In more complex models, other layer types will be declared,
            #       ones that do not require activation intermediates (eg. 
            #       batch normalisation). Hence, by pass activation by passing
            #       an identity function instead.
            layer_activation = params['activation']
            self.layers[layer_name] = torch_parser.parse_activation(
                layer_activation
            )

    ###########
    # Helpers #
    ###########

    @staticmethod
    def __construct_layer_name(layer_idx: int, layer_type: str) -> str:
        """ This function was created as a means for formatting the layer name
            to facilitate finding & handling special cases during forward
            propagation

        Args:
            layer_idx (int): Index of the layer
            layer_type (str): Type of layer
        Returns:
            layer name (str)
        """
        return f"nnl_{layer_idx}_{layer_type.lower()}" 


    @staticmethod
    def __parse_layer_name(layer_name: str) -> Tuple[str, str]:
        """ This function was created as a means for reversing the formatting
            done during name creation to facilitate finding & handling special 
            cases during forward propagation

        Args:
            layer name (str)
        Returns:
            layer_idx (int): Index of the layer
            layer_type (str): Type of layer
        """
        _, layer_idx, layer_type = layer_name.split('_')
        return layer_idx, layer_type.capitalize()

    ##################
    # Core Functions #
    ##################

    def forward(self, x):
        
        # Apply the appropiate activation functions
        for layer_name, a_func in self.layers.items():
            curr_layer = getattr(self, layer_name)

            _, layer_type = self.__parse_layer_name(layer_name)

            # Check if current layer is a recurrent layer
            if layer_type in self.__SPECIAL_CASES:
                x, _ = a_func(curr_layer(x))
            else:
                x = a_func(curr_layer(x))

        return x



#########################################
# Model Communication Class - ModelPlan #
#########################################

class ModelPlan(sy.Plan):
    """
    The Model class serves to automate the building of structured deep neural
    nets, given specific layer configurations. Being a parent class of sy.Plan,
    this makes it more efficient to deploy in terms of communication costs.

    Args:
        owner (VirtualWorker/WebsocketClientWorker): Handler of this model
        structure (OrderedDict): Configurations used to build the achitecture of the NN
        is_condensed (bool): Toggles Binary or Multiclass prediction

    Attributes:
        is_condensed  (bool): Toggles Binary or Multiclass prediction
        layers (OrderedDict): Maps specific layers to their respective activations
        + <Specific layer configuration dynamically defined>
    """
    def __init__(self, structure):
        super().__init__()
        self.__SPECIAL_CASES = ['RNNBase', 'RNN', 'RNNCell', 
                                'LSTM', 'LSTMCell',
                                'GRU', 'GRUCell']
        
        self.layers = OrderedDict()

        for layer, params in enumerate(structure):

            # Detect if input layer
            is_input_layer = params['is_input']

            # Detect layer type
            layer_type = params['l_type']

            # Construct layer name (eg. nnl_0_linear)
            layer_name = self.__construct_layer_name(layer, layer_type)

            # Extract layer structure and initialise layer
            layer_structure = params['structure']
            setattr(
                self, 
                layer_name,
                torch_parser.parse_layer(layer_type)(**layer_structure)
            )

            # Detect activation function & store it for use in .forward()
            # Note: In more complex models, other layer types will be declared,
            #       ones that do not require activation intermediates (eg. 
            #       batch normalisation). Hence, by pass activation by passing
            #       an identity function instead.
            layer_activation = params['activation']
            self.layers[layer_name] = torch_parser.parse_activation(
                layer_activation
            )

    ###########
    # Helpers #
    ###########

    @staticmethod
    def __construct_layer_name(layer_idx: int, layer_type: str) -> str:
        """ This function was created as a means for formatting the layer name
            to facilitate finding & handling special cases during forward
            propagation

        Args:
            layer_idx (int): Index of the layer
            layer_type (str): Type of layer
        Returns:
            layer name (str)
        """
        return f"nnl_{layer_idx}_{layer_type.lower()}" 


    @staticmethod
    def __parse_layer_name(layer_name: str) -> Tuple[str, str]:
        """ This function was created as a means for reversing the formatting
            done during name creation to facilitate finding & handling special 
            cases during forward propagation

        Args:
            layer name (str)
        Returns:
            layer_idx (int): Index of the layer
            layer_type (str): Type of layer
        """
        _, layer_idx, layer_type = layer_name.split('_')
        return layer_idx, layer_type.capitalize()

    ##################
    # Core Functions #
    ##################

    def forward(self, x):
        
        # Apply the appropiate activation functions
        for layer_name, a_func in self.layers.items():
            curr_layer = getattr(self, layer_name)

            _, layer_type = self.__parse_layer_name(layer_name)

            # Check if current layer is a recurrent layer
            if layer_type in self.__SPECIAL_CASES:
                x, _ = a_func(curr_layer(x))
            else:
                x = a_func(curr_layer(x))

        return x


    def build(self, shape):
        """ Uses a declared shape to create mock data for building the 
            customised plan

        Args:
            shape (tuple):
        Returns:

        """
        mock_data = th.rand(shape)
        return super().build(mock_data)


#########
# Tests #
#########

if __name__ == "__main__":

    from pprint import pprint

    model_structure = [
        {
            "activation": "sigmoid",
            "is_input": True,
            "l_type": "Linear",
            "structure": {
                "bias": True,
                "in_features": 15,
                "out_features": 100
            }
        },
        {
            "activation": "sigmoid",
            "is_input": False,
            "l_type": "Linear",
            "structure": {
                "bias": True,
                "in_features": 100,
                "out_features": 90
            }
        },
        {
            "activation": "sigmoid",
            "is_input": False,
            "l_type": "Linear",
            "structure": {
                "bias": True,
                "in_features": 90,
                "out_features": 80
            }
        },
        {
            "activation": "sigmoid",
            "is_input": False,
            "l_type": "Linear",
            "structure": {
                "bias": True,
                "in_features": 80,
                "out_features": 70
            }
        },
        {
            "activation": "sigmoid",
            "is_input": False,
            "l_type": "Linear",
            "structure": {
                "bias": True,
                "in_features": 70,
                "out_features": 60
            }
        },
        {
            "activation": "sigmoid",
            "is_input": False,
            "l_type": "Linear",
            "structure": {
                "bias": True,
                "in_features": 60,
                "out_features": 50
            }
        },
        {
            "activation": "sigmoid",
            "is_input": False,
            "l_type": "Linear",
            "structure": {
                "bias": True,
                "in_features": 50,
                "out_features": 40
            }
        },
        {
            "activation": "sigmoid",
            "is_input": False,
            "l_type": "Linear",
            "structure": {
                "bias": True,
                "in_features": 40,
                "out_features": 30
            }
        },
        {
            "activation": "sigmoid",
            "is_input": False,
            "l_type": "Linear",
            "structure": {
                "bias": True,
                "in_features": 30,
                "out_features": 20
            }
        },
        {
            "activation": "sigmoid",
            "is_input": False,
            "l_type": "Linear",
            "structure": {
                "bias": True,
                "in_features": 20,
                "out_features": 10
            }
        },
        {
            "activation": "sigmoid",
            "is_input": False,
            "l_type": "Linear",
            "structure": {
                "bias": True,
                "in_features": 10,
                "out_features": 1
            }
        }
    ]

    # model_structure = [
    #     # Input: N, C, Height, Width [N, 1, 28, 28]
    #     {
    #         "activation": "relu",
    #         "is_input": True,
    #         "l_type": "Conv2d",
    #         "structure": {
    #             "in_channels": 1, 
    #             "out_channels": 4, # [N, 4, 28, 28]
    #             "kernel_size": 3,
    #             "stride": 1,
    #             "padding": 1
    #         }
    #     },
    #     {
    #         "activation": None,
    #         "is_input": False,
    #         "l_type": "Flatten",
    #         "structure": {}
    #     },
    #     # ------------------------------
    #     {
    #         "activation": "sigmoid",
    #         "is_input": False,
    #         "l_type": "Linear",
    #         "structure": {
    #             "bias": True,
    #             "in_features": 4 * 28 * 28,
    #             "out_features": 1
    #         }
    #     }
    # ]

    hook = sy.TorchHook(th)
    bob = sy.VirtualWorker(hook, id='bob')
    
    # model = Model(model_structure)
    # pprint(model.__dict__)
    # pprint(model.state_dict())

    model_plan = Model(structure=model_structure)
    print(model_plan.include_state)
    print(model_plan.layers)

    print("-->", model_plan.build(shape=(1, 15)))
    # print("-->", model_plan.build(shape=(1, 1, 28, 28)))

    ###########################################################################
    # Class Model can be built! No need for forward() to be the only function #
    ###########################################################################
    
    # print("Before sending:", list(model_plan.parameters()))
    # model_plan_ptr = model_plan.send(bob)
    # print("After sending:", model_plan_ptr.location, list(model_plan_ptr.parameters()))
    # print("Can load_data_dict()?", model_plan.load_state_dict)
    
    # train_data = th.rand([100,15]).send(bob)
    # print(model_plan_ptr(train_data).get())

    ###############################################
    # Copies of un-built plans cannot be built... #
    ###############################################

    # model_plan_copy = model_plan.copy()
    # print(f"Before building: {model_plan_copy}")
    # model_plan_copy.build()
    # print(f"After building: {model_plan_copy}")

    #######################################
    # Built plans can be copied and used! #
    #######################################

    model_plan_copy = model_plan.copy()
    print(f"Before building: {model_plan_copy}")
    print(f"After building: {model_plan_copy}")

    print("Before sending:", list(model_plan_copy.parameters()))
    model_plan_ptr = model_plan_copy.send(bob)
    print("After sending:", model_plan_ptr.location, list(model_plan_ptr.parameters()))

    train_data = th.rand((32, 15)).send(bob)
    print(model_plan_ptr(train_data).get(), model_plan_ptr(train_data).shape)


