#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
from typing import List, Dict

# Libs
import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker

# Custom
from synalgo.arguments import Arguments
from synalgo.model import Model
from synalgo.algorithms.base import BaseAlgorithm

##################
# Configurations #
##################


######################################
# Federated Algorithm Class - FedAvg #
######################################

class FedAvg(BaseAlgorithm):
    """ 
    Implements the fundamental federated averaging algorithm.

    Attributes:
        action (str): Type of ML operation to be executed. Supported options
            are as follows:
            1) 'regress': Orchestrates FL grid to perform regression
            2) 'classify': Orchestrates FL grid to perform classification
            3) 'cluster': TBA
            4) 'associate': TBA
        crypto_provider (VirtualWorker): Trusted Third Party coordinating FL
        workers (list(WebsocketClientWorker)): All particiating CLIENT workers
        arguments (Arguments): Arguments to be passed into each FL function
        train_loader (sy.FederatedLoader): Training data in configured batches
        eval_loader (sy.FederatedLoader): Validation data in configured batches
        test_loader (sy.FederatedLoader): Testing data in configured batches
        global_model (Model): Federatedly-trained Global model
        local_models (dict(str, Models)): Most recent cache of local models
        loss_history (dict): Local & global losses tracked throughout FL cycle
        out_dir (str): Output directory for exporting models & metrics
        checkpoints (dict): All checkpointed models & metrics accumulated
    """
    
    def __init__(
        self, 
        action: str,
        crypto_provider: sy.VirtualWorker,
        workers: List[WebsocketClientWorker],
        arguments: Arguments,
        train_loader: sy.FederatedDataLoader,
        eval_loader: sy.FederatedDataLoader,
        test_loader: sy.FederatedDataLoader,
        global_model: Model,
        local_models: Dict[str, Model] = {},
        out_dir: str = '.',
    ):
        # For custom criterion to behave like the vanilla fedavg, the 
        # coefficient mu MUST be 0 (i.e. No FedProx effect)
        # enter logging statement:  initialize fedavg algo 
        assert self.arguments.mu == 0.0

        super().__init__(
            action=action,
            crypto_provider=crypto_provider,
            workers=workers,
            arguments=arguments,
            train_loader=train_loader,
            eval_loader=eval_loader,
            test_loader=test_loader,
            global_model=global_model,
            local_models=local_models,
            out_dir=out_dir
        )

    ##################
    # Core functions #
    ##################

    def analyse(self):
        """ Calculates contributions of all workers towards the final global 
            model. 
        """
        raise NotImplementedError

