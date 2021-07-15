#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import asyncio
import copy
import inspect
import json
import logging
import os
from collections import OrderedDict
from logging import NOTSET
from multiprocessing import Manager
from pathlib import Path
from typing import Tuple, List, Dict, Union

# Libs
import syft as sy
import torch as th
from sklearn.metrics import (
    accuracy_score, 
    roc_curve,
    roc_auc_score, 
    auc, 
    precision_recall_curve, 
    precision_score,
    recall_score,
    f1_score, 
    confusion_matrix
)
from sklearn.metrics.cluster import contingency_matrix
from syft.workers.websocket_client import WebsocketClientWorker
from tqdm import tqdm

# Custom
from .abstract import AbstractAlgorithm
from synalgo.interfaces import Arguments, EarlyStopping, Model

##################
# Configurations #
##################

SOURCE_FILE = os.path.abspath(__file__)

##################################################
# Federated Algorithm Base Class - BaseAlgorithm #
##################################################

class BaseAlgorithm(AbstractAlgorithm):
    """ 
    Contains baseline functionality to all algorithms. Other specific 
    algorithms will inherit all functionality for handling basic federated
    mechanisms. Extensions of this class will need to override 5 key methods 
    (i.e. `fit`, `evaluate`, `analyse`, `export`, `restore`)

    IMPORTANT:
    This class SHOULD NOT be instantiated by itself! instead, use it to subclass
    other algorithms.

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
        **kwargs
    ):
        # General attributes
        self.action = action

        # Network attributes
        self.crypto_provider = crypto_provider
        self.workers = workers

        # Data attributes
        self.arguments = arguments
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        
        # Model attributes
        self.global_model = global_model
        self.local_models = local_models
        self.loss_history = {
            'global': {
                'train': {},
                'evaluate': {}
            },
            'local': {}
        }

        # Optimisation attributes
        self.loop = None

        # Export Attributes
        self.out_dir = out_dir
        self.checkpoints = {}

        # Avoid Pytorch deadlock issues
        th.set_num_threads(1)

    ############
    # Checkers #
    ############


    ###########    
    # Helpers #
    ###########

    def build_custom_criterion(self):
        """ Augments a selected criterion with the ability to use FedProx

        Returns:
            Surrogate criterion (SurrogateCriterion)
        """
        CRITERION_NAME = self.arguments.criterion.__name__
        ACTION = self.action

        class SurrogateCriterion(self.arguments.criterion):
            """ A wrapper class to augment a specified PyTorch criterion to 
                suppport level 2 algorithm(s) - FedProx & variants 
            
            Args:
                mu (float): Regularisation term for gamma-inexact minimizer
                l1_lambda (float): Regularisation term for L1 regularisation
                l2_lambda (float): Regularisation term for L2 regularisation
                **kwargs: Keyword arguments to pass to parent criterion
                
            Attributes:
                mu (float): Regularisation term for gamma-inexact minimizer
            """
            def __init__(self, mu, l1_lambda, l2_lambda, **kwargs):
                super(SurrogateCriterion, self).__init__(**kwargs)
                self.__temp = [] # tracks minibatches
                self._cache = [] # tracks epochs
                self.mu = mu
                self.l1_lambda = l1_lambda
                self.l2_lambda = l2_lambda

            def format_params(self, outputs: th.Tensor, labels: th.Tensor) -> th.Tensor:
                """ Casts specified label tensors into an appropriate form
                    compatible with the specified base criterion

                Args:
                    labels (th.Tensor): Target values used for loss calculation
                Returns:
                    Restructured labels (th.Tensor)
                """
                ###########################
                # Implementation Footnote #
                ###########################

                # [Cause]
                # Most criterions, if not all, can be split into 2 distinct 
                # groups; 1 group requires both outputs & targets to have the
                # exact same structure (i.e. (N,*)), while the other requires
                # that the outputs (i.e. (N,C) or (N,D)) be different from 
                # targets (i.e. (N,)). This is the result of specialised
                # multiclass criterion as opposed to standard all purpose
                # criterions.

                # [Problems]
                # Without explicitly stating the machine learning action to be 
                # executed, labels are not handled properly, either being 
                # inappropriately expanded/compressed, or not expressed in the 
                # correct datatype/form. This results in criterion errors raised
                # during the federated cycle

                # [Solution]
                # Add a parameter that explicitly specifies the machine learning 
                # operation to be handled, and take the appropriate action. If
                # labels handled are for a regression problem, it is assumed 
                # that its (N,) structure is already correct -> No need for
                # reformatting. However, if labels handled are instead for a
                # classification problem, then labels for binary classification
                # will remain unchanged (since outputs are (N,)), but labels
                # for multiclass classification must be OHE-ed.

                # Supported Losses
                N_STAR_FORMAT = [
                    "L1Loss", "MSELoss", "PoissonNLLLoss", "KLDivLoss",
                    "BCELoss", "BCEWithLogitsLoss", "SmoothL1Loss"
                ]
                N_C_N_FORMAT = ["CrossEntropyLoss", "NLLLoss"]
                STAR_FORMAT = ["HingeEmbeddingLoss", "SoftMarginLoss"]

                # Unsupported Losses
                N_C_N_C_FORMAT = [
                    "MultiLabelMarginLoss", 
                    "MultiLabelSoftMarginLoss"
                ]
                N_D_N_FORMAT = [
                    "MarginRankingLoss"
                ]
                MISC_FORMAT = [
                    "CosineEmbeddingLoss", "TripletMarginLoss", "CTCLoss"
                ]

                logging.debug(
                    f"Action-related metadata tracked.",
                    action=ACTION,
                    output_shape=outputs.shape,
                    label_shape=labels.shape,
                    ID_path=SOURCE_FILE,
                    ID_class=SurrogateCriterion.__name__,
                    ID_function=SurrogateCriterion.format_params.__name__
                )
                
                if CRITERION_NAME in MISC_FORMAT + N_D_N_FORMAT:
                    logging.error(
                        "ValueError: Specified criterion is currently not supported!", 
                        ID_path=SOURCE_FILE,
                        ID_class=SurrogateCriterion.__name__,
                        ID_function=SurrogateCriterion.format_params.__name__
                    )
                    raise ValueError("Specified criterion is currently not supported!")
                
                elif (
                    (ACTION == "classify" and outputs.shape[-1] > 1) and
                    (CRITERION_NAME not in N_C_N_FORMAT)
                ):
                    # One-hot encode predicted labels
                    ohe_labels = th.nn.functional.one_hot(
                        labels,
                        num_classes=outputs.shape[-1] # assume labels max dim = 2
                    )
                    formatted_labels = (
                        ohe_labels
                        if CRITERION_NAME in N_C_N_C_FORMAT 
                        else ohe_labels.float()
                    )
                    return outputs, formatted_labels    # [(N,*), (N,*)]
                else:
                    # Labels are loaded as (N,1) by default in worker
                    return outputs, labels  # [(N,1),(N,1)] or [(N,*),(N,1)]

            def forward(self, outputs, labels, w, wt):
                # Format labels into criterion-compatible
                formatted_outputs, formatted_labels = self.format_params(
                    outputs=outputs,
                    labels=labels
                )
    
                logging.debug(
                    f"Label metadata tracked.",
                    label_shape=labels.shape,
                    label_type=labels.type(), 
                    ID_path=SOURCE_FILE,
                    ID_class=SurrogateCriterion.__name__,
                    ID_function=SurrogateCriterion.forward.__name__
                )
                logging.debug(
                    f"Formatted label metadata tracked.", 
                    formatted_label_shape=formatted_labels.shape,
                    formatted_label_type=formatted_labels.type(), 
                    ID_path=SOURCE_FILE,
                    ID_class=SurrogateCriterion.__name__,
                    ID_function=SurrogateCriterion.forward.__name__
                )
                
                # Calculate normal criterion loss
                loss = super().forward(formatted_outputs, formatted_labels)
                
                logging.debug(
                    f"Location of criterion Loss tracked.",
                    loss_location=loss.location, 
                    ID_path=SOURCE_FILE,
                    ID_class=SurrogateCriterion.__name__,
                    ID_function=SurrogateCriterion.forward.__name__
                )

                # Calculate regularisation terms
                # Note: All regularisation terms have to be collected in some 
                #       iterable first before summing up because in-place 
                #       operation break PyTorch's computation graph
                fedprox_reg_terms = []
                l1_reg_terms = []
                l2_reg_terms = []
                for layer, layer_w in w.items():
                    
                    # Extract corresponding global layer weights
                    layer_wt = wt[layer]

                    # Note: In syft==0.2.4, 
                    # 1) `th.norm(<PointerTensor>)` will always return 
                    #    `tensor(0.)`, hence the need to manually apply the 
                    #    regularisation formulas. However, in future versions 
                    #    when this issue is solved, revert back to cleaner 
                    #    implementation using `th.norm`.

                    # Calculate FedProx regularisation
                    """ 
                    [REDACTED in syft==0.2.4]
                    norm_diff = th.norm(layer_w - layer_wt)
                    fp_reg_term = self.mu * 0.5 * (norm_diff**2)
                    """
                    norm_diff = th.pow((layer_w - layer_wt), 2).sum()
                    fp_reg_term = self.mu * 0.5 * norm_diff # exp cancelled out
                    fedprox_reg_terms.append(fp_reg_term)
                    
                    # Calculate L1 regularisation
                    """
                    [REDACTED in syft==0.2.4]
                    l1_norm = th.norm(layer_w, p=1)
                    l1_reg_term = self.l1_lambda * l1_norm
                    """
                    l1_norm = layer_w.abs().sum()
                    l1_reg_term = self.l1_lambda * l1_norm
                    l1_reg_terms.append(l1_reg_term)
                    
                    # Calculate L2 regularisation
                    """
                    [REDACTED in syft==0.2.4]
                    l2_norm = th.norm(layer_w, p=2)
                    l2_reg_term = self.l2_lambda * 0.5 * (l2_norm)**2
                    """
                    l2_norm = th.pow(layer_w, 2).sum()
                    l2_reg_term = self.l2_lambda * 0.5 * l2_norm
                    l2_reg_terms.append(l2_reg_term)
                
                # Summing up from a list instead of in-place changes 
                # prevents the breaking of the autograd's computation graph
                fedprox_loss = th.stack(fedprox_reg_terms).sum()
                l1_loss = th.stack(l1_reg_terms).sum()
                l2_loss = th.stack(l2_reg_terms).sum()

                # Add up all losses involved
                surrogate_loss = loss + fedprox_loss + l1_loss + l2_loss

                # Store result in cache
                self.__temp.append(surrogate_loss)
                
                return surrogate_loss

            def log(self):
                """ Computes mean loss across all current runs & caches the result """
                avg_loss = th.mean(th.stack(self.__temp), dim=0)
                self._cache.append(avg_loss)
                self.__temp.clear()
                return avg_loss
            
            def reset(self):
                self.__temp = []
                self._cache = []
                return self

        return SurrogateCriterion


    def generate_local_models(self) -> Dict[WebsocketClientWorker, sy.Plan]:
        """ Abstracts the generation of local models in a federated learning
            context. For default FL training (i.e. non-SNN/FedAvg/Fedprox),
            local models generated are clones of the previous round's global
            model. Conversely, in SNN, the local models are instances of
            participant-specified models with supposedly pre-optimised
            architectures.

            IMPORTANT: 
            DO NOT distribute models (i.e. .send()) to local workers. Sending &
            retrieval have to be handled in the same functional context, 
            otherwise PySyft will have a hard time cleaning up residual tensors.

        Returns:
            Distributed context-specific local models (dict(str, Model))
        """
        return {w: self.global_model.copy() for w in self.workers}


    def perform_parallel_training(
        self,
        datasets: dict, 
        models: dict, 
        cache: dict, 
        optimizers: dict, 
        schedulers: dict, 
        criterions: dict, 
        stoppers: dict, 
        rounds: int,
        epochs: int
    ):
        """ Parallelizes training across each distributed dataset 
            (i.e. simulated worker) Parallelization here refers to the 
            training of all distributed models per epoch.
            Note: All objects involved in this set of operations have
                already been distributed to their respective workers

            Parallelization is done across a dataloader of this structure:
            
            [
                # Batch 1
                {
                    worker_1: (data_ptr, label_ptr),
                    worker_2: (data_ptr, label_ptr),
                    ...
                },

                # Batch 2,
                # Batch 3,
                ...
            ]

        Args:
            datasets   (dict(DataLoader)): Distributed training datasets
            models     (dict(nn.Module)): Local models
            cache      (dict(nn.Module)): Cached models from previous rounds
            optimizers (dict(th.optim)): Local optimizers
            schedulers (dict(lr_scheduler)): Local LR schedulers
            criterions (dict(th.nn)): Custom local objective function
            stoppers   (dict(EarlyStopping)): Local early stopping drivers
            rounds (int): Current round of training
            epochs (int): No. of epochs to train each local model
        Returns:
            trained local models
        """ 
        # Tracks which workers have reach an optimal/stagnated model
        # WORKERS_STOPPED = Manager().list()
        WORKERS_STOPPED = []

        async def train_worker(packet):
            """ Train a worker on its single batch, and does an in-place 
                updates for its local model, optimizer & criterion 
            
            Args:
                packet (dict):
                    A single packet of data containing the worker and its
                    data to be trained on 

            """ 
            worker, (data, labels) = packet

            logging.log(
                level=NOTSET,
                msg="Data & labels tracked.",
                data=data,
                labels=labels,
                ID_path=SOURCE_FILE,
                ID_function=train_worker.__name__
            )
            logging.debug(
                "Data & label metadata tracked.",
                data_type=type(data),
                data_shape=data.shape,
                label_type=type(labels),
                label_shape=labels.shape,
                ID_path=SOURCE_FILE,
                ID_function=train_worker.__name__
            )

            # Extract essentials for training
            curr_global_model = cache[worker]
            curr_local_model = models[worker]
            curr_optimizer = optimizers[worker]
            curr_criterion = criterions[worker]

            # Check if worker has been stopped
            if worker.id not in WORKERS_STOPPED:

                # curr_global_model = self.secret_share(curr_global_model)
                # curr_local_model = self.secret_share(curr_local_model)
                curr_global_model = curr_global_model.send(worker)
                curr_local_model = curr_local_model.send(worker)

                logging.debug(
                    f"Location of global model: {curr_global_model.location}", 
                    ID_path=SOURCE_FILE,
                    ID_function=train_worker.__name__
                )
                logging.debug(
                    f"Location of local model: {curr_local_model.location}", 
                    ID_path=SOURCE_FILE,
                    ID_function=train_worker.__name__
                )
                logging.debug(
                    f"Location of X & y: {data.location} {labels.location}", 
                    ID_path=SOURCE_FILE,
                    ID_function=train_worker.__name__
                )

                # Zero gradients to prevent accumulation  
                curr_local_model.train()
                curr_optimizer.zero_grad() 

                # Forward Propagation
                outputs = curr_local_model(data)

                logging.debug(
                    f"Data shape: {data.shape}", 
                    ID_path=SOURCE_FILE,
                    ID_function=train_worker.__name__
                )
                logging.debug(
                    f"Output size: {outputs.shape}", 
                    ID_path=SOURCE_FILE,
                    ID_function=train_worker.__name__
                )
                logging.debug(
                    f"Augmented labels size: {labels.shape}", 
                    ID_path=SOURCE_FILE,
                    ID_function=train_worker.__name__
                )

                loss = curr_criterion(
                    outputs=outputs, 
                    labels=labels,
                    w=curr_local_model.state_dict(),
                    wt=curr_global_model.state_dict()
                )

                # Backward propagation
                loss.backward()
                curr_optimizer.step()

                curr_global_model = curr_global_model.get()
                curr_local_model = curr_local_model.get()

            # Update all involved objects
            assert models[worker] is curr_local_model
            assert optimizers[worker] is curr_optimizer
            assert criterions[worker] is curr_criterion

        async def train_batch(batch):
            """ Asynchronously train all workers on their respective 
                allocated batches 

            Args:
                batch (dict): 
                    A single batch from a sliced dataset stratified by
                    workers and their respective packets. A packet is a
                    tuple pairing of the worker and its data slice
                    i.e. (worker, (data, labels))
            """
            for worker_future in asyncio.as_completed(
                map(train_worker, batch.items())
            ):
                await worker_future

        async def check_for_stagnation(worker):
            """ After a full epoch, check if training for worker has 
                stagnated

            Args:
                worker (WebsocketServerWorker): Worker to be evaluated
            """
            # Extract essentials for adaptation
            curr_local_model = models[worker]
            curr_criterion = criterions[worker]
            curr_scheduler = schedulers[worker]
            curr_stopper = stoppers[worker]

            # Check if worker has been stopped
            if worker.id not in WORKERS_STOPPED:

                # Retrieve final loss computed for this epoch for evaluation
                final_batch_loss = curr_criterion.log()
                curr_stopper(final_batch_loss, curr_local_model)

                # If model is deemed to have stagnated, stop training
                if curr_stopper.early_stop:
                    WORKERS_STOPPED.append(worker.id)
                    
                # else, perform learning rate decay
                else:

                    ###########################
                    # Implementation Footnote #
                    ###########################

                    # [Causes]
                    # Any learning rate scheduler with "plateau" 
                    # (eg. "ReduceLROnPlateau") requires 'self.metric' to be 
                    # passed as a parameter in '.step(...)'

                    # [Problems]
                    # Without this parameter, the following error will be raised:
                    # "TypeError: step() missing 1 required positional argument: 'metrics'"
                    
                    # [Solution]
                    # Check parameters of 'schedueler.step()'. Try if scheduler 
                    # is affected, if scheduler is not compatible, return to 
                    # default call.

                    step_args = list(inspect.signature(curr_scheduler.step).parameters)

                    if 'metrics' in step_args:
                        curr_scheduler.step(final_batch_loss)
                    else:
                        curr_scheduler.step()
                    
            assert schedulers[worker] is curr_scheduler
            assert stoppers[worker] is curr_stopper 

        async def train_datasets(datasets):
            """ Train all batches in a composite federated dataset """
            # Note: All TRAINING must be synchronous w.r.t. each batch, so
            #       that weights can be updated sequentially!
            for batch in datasets:
                await train_batch(batch)
            
            logging.debug(
                f"Before stagnation evaluation - Workers stopped tracked.",
                workers_stopped=WORKERS_STOPPED,
                ID_path=SOURCE_FILE,
                ID_function=train_datasets.__name__
            )

            stagnation_futures = [
                check_for_stagnation(worker) 
                for worker in self.workers
            ]
            await asyncio.gather(*stagnation_futures)

            logging.debug(
                f"After stagnation evaluation - Workers stopped tracked.",
                workers_stopped=WORKERS_STOPPED,
                ID_path=SOURCE_FILE,
                ID_function=train_datasets.__name__
            )

        # computer --> train_dataset --> train_batch --> train_worker

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            for epoch in range(epochs):

                asyncio.get_event_loop().run_until_complete(
                    train_datasets(datasets=datasets)
                )

                # Update cache for local models
                self.local_models = {w.id:lm for w,lm in models.items()}

                # Export ONLY local models. Losses will be accumulated and
                # cached. This is to prevent the autograd computation graph
                # from breaking and interfering with weight updates
                round_key = f"round_{rounds}"
                epoch_key = f"epoch_{epoch}"
                checkpoint_dir = os.path.join(
                    self.out_dir, 
                    "checkpoints",
                    round_key, 
                    epoch_key
                )
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                grid_checkpoint = self.export(
                    out_dir=checkpoint_dir,
                    excluded=['checkpoint']
                )
                for _, logs in grid_checkpoint.items():
                    origin = logs.pop('origin')

                    # Note: Structure - {worker: {round: {epoch: {...}}}}
                    worker_archive = self.checkpoints.get(origin, {}) 
                    round_archive = worker_archive.get(round_key, {}) 
                    round_archive.update({epoch_key: logs})           
                    worker_archive.update({round_key: round_archive})
                    self.checkpoints.update({origin: worker_archive})

        finally:
            loop.close()

        return models, optimizers, schedulers, criterions, stoppers

    
    def calculate_global_params(self, global_model, models, datasets):
        """ Aggregates weights from locally trained models after a round.

            Note: 
                This is based on the assumption that querying database size 
                does not break FL abstraction (i.e. unwilling to share 
                quantity)

        Args:
            global_model (nn.Module): Global model to be trained federatedly
            models (dict(nn.Module)): Trained local models
            datasets (dict(sy.FederatedDataLoader)): Distributed datasets
        Returns:
            Aggregated parameters (OrderedDict)
        """
        global_model.eval()
        for _, local_model in models.items():
            local_model.eval()

        param_types = global_model.state_dict().keys()
        model_states = {w: m.state_dict() for w,m in models.items()}

        # Find size of all distributed datasets for computing scaling factor
        obs_counts = {}
        for batch in datasets:
            for worker, (data, _) in batch.items():
                obs_counts[worker] = obs_counts.get(worker, 0) + len(data)

        # Calculate scaling factors for each worker
        scale_coeffs = {
            worker: local_count/sum(obs_counts.values()) 
            for worker, local_count in obs_counts.items()
        }

        # PyTorch models can only swap weights of the same structure. Hence,
        # aggregate weights while maintaining original layering structure
        aggregated_params = OrderedDict()
        for p_type in param_types:

            param_states = [
                th.mul(
                    model_states[w][p_type],
                    scale_coeffs[w]
                ) for w in self.workers
            ]

            layer_shape = tuple(global_model.state_dict()[p_type].shape)

            # logging.warning(f"{p_type} - Layer shape: {layer_shape}\n{global_model.state_dict()[p_type]}")

            if layer_shape:

                aggregated_params[p_type] = th.stack(
                    param_states,
                    dim=0
                ).sum(dim=0).view(*layer_shape) 

        return aggregated_params
 

    def perform_FL_evaluation(
        self, 
        datasets: Tuple[sy.PointerTensor], 
        workers: List[str] = [], 
        is_shared: bool = True, 
        **kwargs
    ): 
        """ Obtains predictions given a validation/test dataset upon 
            a specified trained global model.

            Parallelization is done across a dataloader of this structure:
            
            [
                # Batch 1
                {
                    worker_1: (data_ptr, label_ptr),
                    worker_2: (data_ptr, label_ptr),
                    ...
                },

                # Batch 2,
                # Batch 3,
                ...
            ]

        Args:
            datasets (tuple(th.Tensor)): A validation/test dataset
            workers (list(str)): Filter to select specific workers to infer on
            is_shared (bool): Toggles whether SMPC is turned on
            **kwargs: Miscellaneous keyword arguments for future use
        Returns:
            Tagged prediction tensor (sy.PointerTensor)
        """

        async def evaluate_worker(packet):
            """ Evaluate a worker on its single packet of minibatch data
            
            Args:
                packet (dict):
                    A single packet of data containing the worker and its
                    data to be evaluated upon
            """ 
            logging.log(
                level=NOTSET,
                msg="Packet tracked.",
                packet=packet,
                ID_path=SOURCE_FILE,
                ID_function=evaluate_worker.__name__
            )

            worker, (data, labels) = packet

            logging.log(
                level=NOTSET,
                msg="Data & labels tracked.",
                data=data,
                labels=labels,
                ID_path=SOURCE_FILE,
                ID_function=evaluate_worker.__name__
            )
            logging.debug(
                "Data & label metadata tracked.",
                data_type=type(data),
                data_shape=data.shape,
                label_type=type(labels),
                label_shape=labels.shape,
                ID_path=SOURCE_FILE,
                ID_function=evaluate_worker.__name__
            )
            logging.debug(
                f"Current worker evaluated tracked.",
                worker=worker, 
                type=type(worker), 
                ID_path=SOURCE_FILE,
                ID_function=evaluate_worker.__name__
            )

            # Skip predictions if filter was specified, and current worker was
            # not part of the selected workers
            if workers and (worker.id not in workers):
                return {}, None

            self.global_model = self.global_model.send(worker)
            self.local_models[worker.id] = self.local_models[worker.id].send(worker)

            self.global_model.eval()
            self.local_models[worker.id].eval()
            with th.no_grad():

                outputs = self.global_model(data).detach()

                if self.action == "regress":
                    # Predictions are the raw outputs
                    pass

                elif self.action == "classify":
                    class_count = outputs.shape[1]
                    # If multi-class, use argmax
                    if class_count > 2:
                        # One-hot encode predicted labels
                        _, predictions = outputs.max(axis=1)
                    else:
                        # For binary, use 0.5 as threshold
                        predictions = (outputs > 0.5).float()

                else:
                    logging.error(
                        f"ValueError: ML action {self.action} is not supported!", 
                        ID_path=SOURCE_FILE,
                        ID_function=evaluate_worker.__name__
                    )
                    raise ValueError(f"ML action {self.action} is not supported!")

                # Compute loss
                surrogate_criterion = self.build_custom_criterion()(
                    **self.arguments.criterion_params
                )

                loss = surrogate_criterion(
                    outputs=outputs, 
                    labels=labels,
                    w=self.local_models[worker.id].state_dict(),
                    wt=self.global_model.state_dict()
                )

            self.local_models[worker.id] = self.local_models[worker.id].get()
            self.global_model = self.global_model.get()

            #############################################
            # Inference V1: Assume TTP's role is robust #
            #############################################
            # In this version, TTP's coordination is not deemed to be breaking
            # FL rules. Hence predictions & labels can be pulled in locally for
            # calculating statistics, before sending the labels back to worker.

            # labels = labels.get()
            # outputs = outputs.get()
            # predictions = predictions.get()

            # logging.debug(f"labels: {labels}, outputs: {outputs}, predictions: {predictions}")

            # # Calculate accuracy of predictions
            # accuracy = accuracy_score(labels.numpy(), predictions.numpy())
            
            # # Calculate ROC-AUC for each label
            # roc = roc_auc_score(labels.numpy(), outputs.numpy())
            # fpr, tpr, _ = roc_curve(labels.numpy(), outputs.numpy())
            
            # # Calculate Area under PR curve
            # pc_vals, rc_vals, _ = precision_recall_curve(labels.numpy(), outputs.numpy())
            # auc_pr_score = auc(rc_vals, pc_vals)
            
            # # Calculate F-score
            # f_score = f1_score(labels.numpy(), predictions.numpy())

            # # Calculate contingency matrix
            # ct_matrix = contingency_matrix(labels.numpy(), predictions.numpy())
            
            # # Calculate confusion matrix
            # cf_matrix = confusion_matrix(labels.numpy(), predictions.numpy())
            # logging.debug(f"Confusion matrix: {cf_matrix}")

            # TN, FP, FN, TP = cf_matrix.ravel()
            # logging.debug(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")

            # # Sensitivity, hit rate, recall, or true positive rate
            # TPR = TP/(TP+FN) if (TP+FN) != 0 else 0
            # # Specificity or true negative rate
            # TNR = TN/(TN+FP) if (TN+FP) != 0 else 0
            # # Precision or positive predictive value
            # PPV = TP/(TP+FP) if (TP+FP) != 0 else 0
            # # Negative predictive value
            # NPV = TN/(TN+FN) if (TN+FN) != 0 else 0
            # # Fall out or false positive rate
            # FPR = FP/(FP+TN) if (FP+TN) != 0 else 0
            # # False negative rate
            # FNR = FN/(TP+FN) if (TP+FN) != 0 else 0
            # # False discovery rate
            # FDR = FP/(TP+FP) if (TP+FP) != 0 else 0

            # statistics = {
            #     'accuracy': accuracy,
            #     'roc_auc_score': roc,
            #     'pr_auc_score': auc_pr_score,
            #     'f_score': f_score,
            #     'TPR': TPR,
            #     'TNR': TNR,
            #     'PPV': PPV,
            #     'NPV': NPV,
            #     'FPR': FPR,
            #     'FNR': FNR,
            #     'FDR': FDR,
            #     'TP': TP,
            #     'TN': TN,
            #     'FP': FP,
            #     'FN': FN
            # }

            # labels = labels.send(worker)

            # return {worker: statistics}

            ####################################################################
            # Inference V1.5: Assume TTP's role is robust, but avoid violation #
            ####################################################################
            # In this version, while TTP's coordination is also not deemed to be
            # breaking FL rules, the goal is to violate the minimum no. of
            # federated procedures. Here, only outputs & predictions can be 
            # pulled in locally, since they are deemed to be TTP-generated.
            # However, statistical calculation will be orchestrated to be done
            # at worker nodes, and be sent back via a flask payload. This way,
            # the TTP avoids even looking at client's raw data, only interacting
            # with derivative information. 

            outputs = outputs.get()
            predictions = (
                predictions.get()
                if self.action == "classify" 
                else outputs # for regression, predictions are raw outputs
            )
            loss = loss.get()

            return {worker: {"y_pred": predictions, "y_score": outputs}}, loss

            ####################################################################
            # Inference V2: Strictly enforce federated procedures in inference #
            ####################################################################

            # Override garbage collection to allow for post-inference tracking
            # data.set_garbage_collect_data(False)
            # labels.set_garbage_collect_data(False)
            # outputs.set_garbage_collect_data(False)
            # predictions.set_garbage_collect_data(False)

            # data_id = data.id_at_location
            # labels_id = labels.id_at_location
            # outputs_id = outputs.id_at_location
            # predictions_id = predictions.id_at_location

            # data = data.get()
            # labels = labels.get()
            # outputs = outputs.get()
            # # predictions = predictions.get()

            # logging.debug(f"Before transfer - Worker: {worker}")

            # worker._send_msg_and_deserialize("register_obj", obj=data.tag("#minibatch"))#, obj_id=data_id)
            # worker._send_msg_and_deserialize("register_obj", obj=labels.tag("#minibatch"))#, obj_id=labels_id)
            # worker._send_msg_and_deserialize("register_obj", obj=outputs.tag("#minibatch"))#, obj_id=outputs_id)
            # worker._send_msg_and_deserialize("register_obj", obj=predictions.tag("#minibatch"))#, obj_id=predictions_id)

            # logging.debug(f"After transfer - Worker: {worker}")

            # inferences = {
            #     worker: {
            #         'data': 1,
            #         'labels': 2,
            #         'outputs': 3,
            #         'predictions': 4 
            #     }
            # }

            # # Convert collection of object IDs accumulated from minibatch 
            # inferencer = Inferencer(inferences=inferences, **kwargs["keys"])
            # # converted_stats = inferencer.infer(reg_records=kwargs["registrations"])
            # converted_stats = await inferencer._collect_all_stats(reg_records=kwargs["registrations"])

            # self.global_model = self.global_model.get()
            # return converted_stats

        async def evaluate_batch(batch):
            """ Asynchronously train all workers on their respective 
                allocated batches 

            Args:
                batch (dict): 
                    A single batch from a sliced dataset stratified by
                    workers and their respective packets. A packet is a
                    tuple pairing of the worker and its data slice
                    i.e. (worker, (data, labels))
            """
            logging.debug(
                "Batch detected.",
                batch=f"{batch}", 
                batch_type=type(batch), 
                ID_path=SOURCE_FILE,
                ID_function=evaluate_batch.__name__
            )

            batch_evaluations = {}
            batch_losses = []

            # logging.warning(f">>> batch: {batch}, type: {type(batch)}")

            # If multiple prediction sets have been declared across all workers,
            # batch will be a dictionary i.e. {<worker_1>: (data, labels), ...}
            if isinstance(batch, dict):

                for worker_future in asyncio.as_completed(
                    map(evaluate_worker, batch.items())
                ):
                    evaluated_worker_batch, loss = await worker_future
                    batch_evaluations.update(evaluated_worker_batch)
                    batch_losses.append(loss)

            # If only 1 prediction set is declared (i.e. only 1 guest present), 
            # batch will be a tuple i.e. (data, label)
            elif isinstance(batch, tuple):
                data, labels = batch

                if data.location != labels.location:
                    logging.error(
                        f"RuntimeError: Feature data and label data are not in the same location!", 
                        ID_path=SOURCE_FILE,
                        ID_function=evaluate_batch.__name__
                    )
                    raise RuntimeError("Feature data and label data are not in the same location!")
                
                packet = (data.location, batch)
                evaluated_worker_batch, loss = await evaluate_worker(packet)
                batch_evaluations.update(evaluated_worker_batch)
                batch_losses.append(loss)

            # logging.warning(f">>> batch evaluations: {batch_evaluations}")
            # logging.warning(f">>> batch losses: {batch_losses}")

            return batch_evaluations, batch_losses

        async def evaluate_datasets(datasets):
            """ Train all batches in a composite federated dataset """
            # Note: Unlike in training, inference does not require any weight
            #       tracking, thus each batch can be processed asynchronously 
            #       as well!
            batch_futures = [evaluate_batch(batch) for batch in datasets]
            all_batch_evaluations = await asyncio.gather(*batch_futures)

            ##########################################################
            # Inference V1: Exercise TTP's role as secure aggregator #
            ##########################################################
            
            # all_worker_stats = {}
            # all_worker_preds = {}

            # for b_count, (batch_evaluations, batch_predictions) in enumerate(
            #     all_batch_evaluations, 
            #     start=1
            # ):
            #     for worker, batch_stats in batch_evaluations.items():

            #         # Manage statistical aggregation
            #         aggregated_stats = all_worker_stats.get(worker.id, {})
            #         for stat, value in batch_stats.items():
                        
            #             if stat in ["TN", "FP", "FN", "TP"]:
            #                 total_val = aggregated_stats.get(stat, 0.0) + value
            #                 aggregated_stats[stat] = total_val

            #             else:
            #                 sliding_stat_avg = (
            #                     aggregated_stats.get(stat, 0.0)*(b_count-1) + value
            #                 ) / b_count
            #                 aggregated_stats[stat] = sliding_stat_avg

            #         all_worker_stats[worker.id] = aggregated_stats

            ####################################################################
            # Inference V1.5: Assume TTP's role is robust, but avoid violation #
            ####################################################################

            all_worker_outputs = {}
            all_losses = []
            for batch_evaluations, batch_losses in all_batch_evaluations:

                for worker, outputs in batch_evaluations.items():

                    aggregated_outputs = all_worker_outputs.get(worker.id, {})
                    for _type, result in outputs.items():

                        aggregated_results = aggregated_outputs.get(_type, [])
                        aggregated_results.append(result)
                        aggregated_outputs[_type] = aggregated_results

                    all_worker_outputs[worker.id] = aggregated_outputs

                all_losses += batch_losses

            # Concatenate all batch outputs for each worker
            all_combined_outputs = {
                worker_id: {
                    _type: th.cat(res_collection, dim=0).numpy().tolist()
                    for _type, res_collection in batch_outputs.items()
                }
                for worker_id, batch_outputs in all_worker_outputs.items()
            }
            
            relevant_losses = [loss for loss in all_losses if loss is not None]
            avg_loss = (
                th.mean(th.stack(relevant_losses), dim=0) 
                if relevant_losses 
                else None
            )

            # for worker, i in all_combined_outputs.items():
                # for _type, res_collection in i.items():
                    # logging.warning(f"{worker} - {_type} -> {res_collection} {len(res_collection)}")

            # logging.warning(f">>> all combined outputs: {all_combined_outputs}, avg loss: {avg_loss}")

            return all_combined_outputs, avg_loss

            ####################################################################
            # Inference V2: Strictly enforce federated procedures in inference #
            ####################################################################

            # for batch_evaluations in all_batch_evaluations:
            #     for worker, batch_obj_ids in batch_evaluations.items():

            #         minibatch_ids = all_worker_stats.get(worker.id, [])
            #         minibatch_ids.append(batch_obj_ids)
            #         all_worker_stats[worker.id] = minibatch_ids


        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            all_combined_outputs, avg_loss = asyncio.get_event_loop().run_until_complete(
                evaluate_datasets(datasets=datasets)
            )

        finally:
            loop.close()

        return all_combined_outputs, avg_loss

    ##################
    # Core Functions #
    ##################

    def initialise(self):
        """ Encapsulates all operations required for Fedprox suppport for all
            subsequent child algorithms
        """
        # Generate K copies of template model, representing local models for
        # each worker in preparation for parallel training, and send them to
        # their designated workers
        # Note: This step is crucial because it is able prevent pointer 
        #       mutation, which comes as a result of copying pointers (refer
        #       to Part 4, section X), specifically if the global pointer 
        #       was copied directly.
        local_models = self.generate_local_models()

        # Model weights from previous round for subsequent FedProx 
        # comparison. Due to certain nuances stated below, they have to be
        # specified here. 
        # Note - In syft==0.2.4: 
        # 1) copy.deepcopy(PointerTensor) causes "TypeError: clone() got an 
        #    unexpected keyword argument 'memory_format'"
        # 2) Direct cloning of dictionary of models causes "TypeError: can't 
        #    pickle module objects"
        prev_models = self.generate_local_models()
        
        optimizers = {
            w: self.arguments.optimizer( 
                **self.arguments.optimizer_params,
                params=model.parameters()
            ) for w, model in local_models.items()
        }

        schedulers = {
            w: self.arguments.lr_scheduler(
                **self.arguments.lr_decay_params,
                optimizer=optimizer
            )
            for w, optimizer in optimizers.items()
        }

        criterions = {
            w: self.build_custom_criterion()(
                **self.arguments.criterion_params
            ) for w,m in local_models.items()
        }
        
        stoppers = {
            w: EarlyStopping(
                **self.arguments.early_stopping_params
            ) for w,m in local_models.items()
        }

        return (
            local_models,
            prev_models, 
            optimizers, 
            schedulers, 
            criterions, 
            stoppers
        )


    def fit(self):
        """ Performs federated training using a pre-specified model as
            a template, across initialised worker nodes, coordinated by
            a ttp node.
            
        Returns:
            Trained global model (Model)
        """
        ###########################
        # Implementation Footnote #
        ###########################

        # However, due to certain PySyft nuances (refer to Part 4, section 1: 
        # Frame of Reference) there is a need to choose a conceptual 
        # representation of the overall architecture. Here, the node agnostic 
        # variant is implemented. Model is stored in the server -> Client 
        # (i.e. 'Me') does not interact with it
        
        # Note: If MPC is requested, global model itself cannot be shared, only 
        # its copies are shared. This is due to restrictions in PointerTensor 
        # mechanics.

        global_val_stopper = EarlyStopping(**self.arguments.early_stopping_params)

        rounds = 0
        pbar = tqdm(total=self.arguments.rounds, desc='Rounds', leave=True)
        while rounds < self.arguments.rounds:

            (
                local_models,
                prev_models, 
                optimizers, 
                schedulers, 
                criterions, 
                stoppers
            ) = self.initialise()
            
            (retrieved_models, _, _, _, _) = self.perform_parallel_training(
                datasets=self.train_loader, 
                models=local_models,
                cache=prev_models,
                optimizers=optimizers, 
                schedulers=schedulers,
                criterions=criterions, 
                stoppers=stoppers,
                rounds=rounds,
                epochs=self.arguments.epochs
            )

            logging.debug(
                f"Round {rounds} - Current global model tracked.", 
                global_model=self.global_model.state_dict(),
                ID_path=SOURCE_FILE,
                ID_class=BaseAlgorithm.__name__,
                ID_function=BaseAlgorithm.fit.__name__
            )

            # Retrieve all models from their respective workers
            aggregated_params = self.calculate_global_params(
                self.global_model, 
                retrieved_models, 
                self.train_loader
            )

            # Update weights with aggregated parameters 
            self.global_model.load_state_dict(aggregated_params)
            
            logging.debug(
                f"Round {rounds} - Updated global model tracked.", 
                updated_global_model=self.global_model.state_dict(),
                ID_path=SOURCE_FILE,
                ID_class=BaseAlgorithm.__name__,
                ID_function=BaseAlgorithm.fit.__name__
            )

            final_local_losses = {
                w.id: c._cache[-1].get()
                for w,c in criterions.items()
            }

            # Store local losses for analysis
            for w_id, loss in final_local_losses.items():
                local_loss_archive = self.loss_history['local'].get(w_id, {})
                local_loss_archive.update({rounds: loss.item()})
                self.loss_history['local'][w_id] = local_loss_archive

            global_train_loss = th.mean(
                th.stack(list(final_local_losses.values())),
                dim=0
            )

            # Validate the global model
            _, evaluation_losses = self.evaluate(metas=['evaluate'])
            global_val_loss = evaluation_losses['evaluate']

            # Store global losses for analysis
            global_loss_archive = self.loss_history['global']
            global_train_losses = global_loss_archive.get('train', {})
            global_train_losses.update({rounds: global_train_loss.item()})
            global_val_losses = global_loss_archive.get('evaluate', {})
            global_val_losses.update({rounds: global_val_loss.item()})
            self.loss_history['global'] = {
                'train': global_train_losses,
                'evaluate': global_val_losses
            }

            # If global model is deemed to have stagnated, stop training
            global_val_stopper(global_val_loss, self.global_model)
            if global_val_stopper.early_stop:
                logging.info(
                    "Global model has stagnated. Training round terminated!",
                    ID_path=SOURCE_FILE,
                    ID_class=BaseAlgorithm.__name__,
                    ID_function=BaseAlgorithm.fit.__name__
                )
                break

            rounds += 1
            pbar.update(1)
        
        pbar.close()

        logging.debug(
            f"Objects in TTP tracked.",
            ttp=self.crypto_provider, 
            object_count=len(self.crypto_provider._objects), 
            ID_path=SOURCE_FILE,
            ID_class=BaseAlgorithm.__name__,
            ID_function=BaseAlgorithm.fit.__name__
        )
        logging.debug(
            f"Objects in sy.local_worker tracked.",
            local_worker=sy.local_worker, 
            object_count=len(sy.local_worker._objects), 
            ID_path=SOURCE_FILE,
            ID_class=BaseAlgorithm.__name__,
            ID_function=BaseAlgorithm.fit.__name__
        )

        return self.global_model, self.local_models


    def evaluate(
        self, 
        metas: List[str] = [], 
        workers: List[str] = []
    ) -> Tuple[Dict[str, Dict[str, th.Tensor]], Dict[str, th.Tensor]]:
        """ Using the current instance of the global model, performs inference 
            on pre-specified datasets.

        Args:
            metas (list(str)): Meta tokens indicating which datasets are to be
                evaluated. If empty (default), all meta datasets (i.e. training,
                validation and testing) will be evaluated
            workers (list(str)): Worker IDs of workers whose datasets are to be
                evaluated. If empty (default), evaluate all workers' datasets. 
        Returns:
            Inferences (dict(worker_id, dict(result_type, th.Tensor)))
            losses (dict(str, th.Tensor))
        """
        DATA_MAP = {
            'train': self.train_loader,
            'evaluate': self.eval_loader,
            'predict': self.test_loader
        }

        # If no meta filters are specified, evaluate all datasets 
        metas = list(DATA_MAP.keys()) if not metas else metas

        # If no worker filter are specified, evaluate all workers
        workers = [w.id for w in self.workers] if not workers else workers

        # logging.warning(f"--> metas: {metas}, workers: {workers}")

        # Evaluate global model using datasets conforming to specified metas
        inferences = {}
        losses = {}
        for meta, dataset in DATA_MAP.items():

            # logging.warning(f"meta: {meta}, dataset: {dataset}")

            if meta in metas:

                worker_meta_inference, avg_loss = self.perform_FL_evaluation(
                    datasets=dataset,
                    workers=workers,
                    is_shared=True
                )

                # inference = worker -> meta -> (y_pred, y_score)
                for worker_id, meta_result in worker_meta_inference.items():

                    worker_results = inferences.get(worker_id, {})
                    worker_results[meta] = meta_result
                    inferences[worker_id] = worker_results

                losses[meta] = avg_loss
        
        return inferences, losses


    def analyse(self):
        """ Calculates contributions of each local model towards the current
            state of the global model

        Returns:
        """
        raise NotImplementedError


    def export(self, out_dir: str = None, excluded: List[str] = []) -> dict:
        """ Snapshots the current state of federated cycle and exports all 
            models to file. A dictionary is produced as a rous

            An archive's structure looks like this:
            {
                'global': {
                    'origin': <crypto_provider ID>,
                    'path': <path(s) to exported final global model(s)>,
                    'loss_history': <path(s) to final global loss history(s)>,
                    'checkpoints': {
                        'round_0': {
                            'epoch_0': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported global model(s)>,
                                'loss_history': <path(s) to global loss history(s)>,
                            },
                            'epoch_1': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported globalmodel(s)>,
                                'loss_history': <path(s) to global loss history(s)>,
                            },
                            ...
                        },
                        'round_1': {
                            'epoch_0': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to global exported model(s)>,
                                'loss_history': <path(s) to global loss history(s)>,
                            },
                            'epoch_1': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported global model(s)>,
                                'loss_history': <path(s) to global loss history(s)>,
                            },
                            ...
                        }
                        ...
                    }
                },
                'local_<idx>': {
                    'origin': <worker ID>,
                    'path': <path(s) to exported final local model(s)>,
                    'loss_history': <path(s) to final local loss history(s)>,
                    'checkpoints': {
                        'round_0': {
                            'epoch_0': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported local model(s)>,
                                'loss_history': <path(s) to local loss history(s)>,
                            },
                            'epoch_1': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported local model(s)>,
                                'loss_history': <path(s) to local loss history(s)>,
                            },
                            ...
                        },
                        'round_1': {
                            'epoch_0': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported local model(s)>,
                                'loss_history': <path(s) to local loss history(s)>,
                            },
                            'epoch_1': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported local model(s)>,
                                'loss_history': <path(s) to local loss history(s)>,
                            },
                            ...
                        }
                        ...
                    }
                },
                ...
            }

        Args:
            out_dir (str): Path to output directory for export
            excluded (list(str)): Federated attributes to skip when exporting.
                Attribute options are as follows:
                1) 'global': Skips current state of the global model
                2) 'local': Skips current states of all local models
                3) 'loss': Skips current state of global & local loss histories
                4) 'checkpoint': Skips all checkpointed metadata
        Returns:
            Archive (dict)
        """
        # Override cached output directory with specified directory if any
        out_dir = out_dir if out_dir else self.out_dir

        def save_global_model():
            if 'global' in excluded: return None
            # Export global model to file
            global_model_out_path = os.path.join(
                out_dir, 
                "global_model.pt"
            )
            # Only states can be saved, since Model is not picklable
            th.save(self.global_model.state_dict(), global_model_out_path)
            return global_model_out_path

        def save_global_losses():
            if 'loss' in excluded: return None
            # Export global loss history to file
            global_loss_out_path = os.path.join(
                out_dir, 
                "global_loss_history.json"
            )
            with open(global_loss_out_path, 'w') as glp:
                logging.debug(
                    "Global Loss History tracked.", 
                    global_loss_history=self.loss_history['global'],
                    ID_path=SOURCE_FILE,
                    ID_class=BaseAlgorithm.__name__,
                    ID_function=BaseAlgorithm.export.__name__
                )
                json.dump(self.loss_history['global'], glp)
            return global_loss_out_path

        def save_worker_model(worker_id, model):
            if 'local' in excluded: return None
            # Export local model to file
            local_model_out_path = os.path.join(
                out_dir, 
                f"local_model_{worker_id}.pt"
            )
            if self.arguments.is_snn:
                # Local models are saved directly to log their architectures
                th.save(model, local_model_out_path)
            else:
                th.save(model.state_dict(), local_model_out_path)
            return local_model_out_path

        def save_worker_losses(worker_id):
            if 'loss' in excluded: return None
            # Export local loss history to file
            local_loss_out_path = os.path.join(
                out_dir, 
                f"local_loss_history_{worker_id}.json"
            )
            with open(local_loss_out_path, 'w') as llp:
                json.dump(self.loss_history['local'].get(worker_id, {}), llp)
            return local_loss_out_path

        out_paths = {}

        # Package global metadata for storage
        out_paths['global'] = {
            'origin': self.crypto_provider.id,
            'path': save_global_model(),
            'loss_history': save_global_losses()
        }
        if 'checkpoint' not in excluded:
            out_paths['global'].update({
                'checkpoints': self.checkpoints.get(self.crypto_provider.id, {})
            })

        for idx, (worker_id, local_model) in enumerate(
            self.local_models.items(), 
            start=1
        ):
            # Package local metadata for storage
            out_paths[f'local_{idx}'] = {
                'origin': worker_id,
                'path': save_worker_model(worker_id, model=local_model),
                'loss_history': save_worker_losses(worker_id),
            }
            if 'checkpoint' not in excluded:
                out_paths[f'local_{idx}'].update({
                    'checkpoints': self.checkpoints.get(worker_id, {})
                })

        return out_paths


    def restore(
        self, 
        archive: dict, 
        version: Tuple[str, str] = None
    ):
        """ Restores model states from a previously archived training run. If 
            version is not specified, then restore the final state of the grid.
            If version is specified, restore the state of all models conforming
            to that version's snapshot.

            An archive's structure looks like this:
            {
                'global': {
                    'origin': <crypto_provider ID>,
                    'path': <path(s) to exported final global model(s)>,
                    'loss_history': <path(s) to final global loss history(s)>,
                    'checkpoints': {
                        'round_0': {
                            'epoch_0': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported global model(s)>,
                                'loss_history': <path(s) to globalloss history(s)>,
                            },
                            'epoch_1': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported globalmodel(s)>,
                                'loss_history': <path(s) to global loss history(s)>,
                            },
                            ...
                        },
                        'round_1': {
                            'epoch_0': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to global exported model(s)>,
                                'loss_history': <path(s) to global loss history(s)>,
                            },
                            'epoch_1': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported global model(s)>,
                                'loss_history': <path(s) to global loss history(s)>,
                            },
                            ...
                        }
                        ...
                    }
                },
                'local_<idx>': {
                    'origin': <worker ID>,
                    'path': <path(s) to exported final local model(s)>,
                    'loss_history': <path(s) to final local loss history(s)>,
                    'checkpoints': {
                        'round_0': {
                            'epoch_0': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported local model(s)>,
                                'loss_history': <path(s) to local loss history(s)>,
                            },
                            'epoch_1': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported local model(s)>,
                                'loss_history': <path(s) to local loss history(s)>,
                            },
                            ...
                        },
                        'round_1': {
                            'epoch_0': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported local model(s)>,
                                'loss_history': <path(s) to local loss history(s)>,
                            },
                            'epoch_1': {
                                'origin': <crypto_provider ID>,
                                'path': <path(s) to exported local model(s)>,
                                'loss_history': <path(s) to local loss history(s)>,
                            },
                            ...
                        }
                        ...
                    }
                },
                ...
            }

        Args:
            archive (dict): Dictionary containing versioned histories of 
                exported filepaths corresponding to the state of models within a
                training cycle
            version (tuple(str)): A tuple where the first index indicates the
                round index and the second the epoch index 
                (i.e. (round_<r_idx>, epoch_<e_idx>))
        """
        for _type, logs in archive.items():

            logging.debug(
                f"Model archival logs for {_type} tracked.",
                logs=logs,
                ID_path=SOURCE_FILE,
                ID_class=BaseAlgorithm.__name__,
                ID_function=BaseAlgorithm.restore.__name__
            )

            archived_origin = logs['origin']

            # Check if exact version of the federated grid was specified
            if version:
                round_idx = version[0]
                epoch_idx = version[1]
                filtered_version = logs['checkpoints'][round_idx][epoch_idx]
                archived_state = th.load(filtered_version['path'])
            
            # Otherwise, load the final state of the grid
            else:
                archived_state = th.load(logs['path'])

            if archived_origin == self.crypto_provider.id:
                self.global_model.load_state_dict(archived_state)

            else:

                ###########################
                # Implementation Footnote #
                ###########################

                # Because local models in SNN will be optimal models owned
                # by the participants themselves, there are 2 ways of 
                # handling model archival - Store the full model, or get 
                # participants to register the architecture & hyperparameter
                # sets of their optimal setup, while exporting the model
                # weights. The former allows models to be captured alongside
                # their architectures, hence removing the need for tracking 
                # additional information unncessarily. However, models 
                # exported this way have limited use outside of REST-RPC, 
                # since they are pickled relative to the file structure of 
                # the package. The latter is the more flexible approach, 
                # since weights will still remain usable even outside the
                # context of REST-RPC, as long as local model architectures,
                # are available.  
                
                if self.arguments.is_snn:
                    archived_model = archived_state
                else:
                    archived_model = copy.deepcopy(self.global_model)
                    archived_model.load_state_dict(archived_state)
                    
                self.local_models[archived_origin] = archived_model     