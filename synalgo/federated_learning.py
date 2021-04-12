#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
import logging
import os
from collections import OrderedDict, defaultdict
from multiprocessing import Manager
from pathlib import Path
from typing import Tuple, List, Dict, Union

# Libs
import syft as sy
import torch as th

# Custom
from .config import seed_everything
from synalgo import algorithms
from synalgo.interfaces import Arguments, EarlyStopping, Model

##################
# Configurations #
##################

infinite_nested_dict = lambda: defaultdict(infinite_nested_dict)

###############################################
# Abstract Training Class - FederatedLearning #
###############################################

class FederatedLearning:
    """
    The main class that coodinates federated training across a PySyft-driven 
    PRIVATE grid. This means that only the crypto-provider can excute and
    remote operations (i.e. 1 static reference point in STAR architecture). This
    class servers as a wrapper around algorithm classes, automating the grid
    setup that is required for federated training.

    Attributes:
        crypto_provider (VirtualWorker): Trusted Third Party coordinating FL
        workers (list(WebsocketClientWorker)): All particiating client workers
        grid (sy.PrivateGridNetwork): A grid to facilitate dataset searches
        _aliases (dict): ID-to-worker mappings for ease of reference
        arguments (Arguments): Arguments to be passed into each FL function
        reference (Model): Reference model whose architecture is to be used in
            subsequent model training or inference. Note that this model is
            merely a template, and that the actual models trained are to be
            retrieved from algorithm objects.
        algorithm (BaseAlgorithm): Algorithm object corresponding to specified 
            algorithm retrieved from arguments.
    """
    
    def __init__(
        self, 
        action: str,
        crypto_provider: sy.VirtualWorker, 
        workers: list, 
        arguments: Arguments,
        reference: Model,
        out_dir: str = '.',
        loop=None
    ):
        # General attributes
        self.action = action

        # Network attributes
        self.crypto_provider = crypto_provider
        self.workers = workers
        self.grid = sy.PrivateGridNetwork(self.crypto_provider, *self.workers)
        self._aliases = {w.id: w for w in self.grid.workers}
        
        # Data attributes
        self.arguments = arguments
        self.reference = reference
        self.algorithm = None

        # Export Attributes
        self.out_dir = out_dir

        # Lock random states within server
        seed_everything(seed=self.arguments.seed)

    ############
    # Checkers #
    ############

    def is_loaded(self):
        """ Checks if environment has already been loaded 
        
        Returns:
            loaded state (bool)
        """
        return self.algorithm is not None

    ###########
    # Helpers #
    ###########

    def secret_share(self, tensor):
        """ Transform to fixed precision and secret share a tensor 
        
        Args:
            tensor (PointerTensor): Pointer to be shared
        Returns:
            MPC-shared pointer tensor (PointerTensor)
        """
        return (
            tensor
            .fix_precision(precision_fractional=self.arguments.precision_fractional)
            .share(
                *self.workers, 
                crypto_provider=self.crypto_provider, 
                requires_grad=True
            )
        )


    def setup_FL_env(
        self, 
        is_shared: bool=False
    ) -> Tuple[Dict[sy.WebsocketClientWorker, sy.BaseDataset], ...]:

        """ Sets up a basic federated learning environment using virtual workers,
            with a allocated arbiter (i.e. TTP) to faciliate in model development
            & utilisation, and deploys datasets to their respective workers
            
        Args:
            is_shared (bool): Toggles if SMPC encryption protocols are active
        Returns:
            train_datasets (dict(sy.BaseDataset))
            eval_datasets  (dict(sy.BaseDataset))
            test_datasets  (dict(sy.BaseDataset))
        """
        
        def convert_to_datasets(*tags):
            """ Takes in tags to query across all workers, and groups X & y 
                pointer tensors into datasets
            
            Args:
                *tags (str): Tags to query on
            Returns:
                datasets (dict(WebsocketClientWorker, sy.BaseDataset))
            """
            # Retrieve Pointer Tensors to remote datasets
            pointers = self.grid.search(*tags)
        
            datasets = {}
            for worker_id, data in pointers.items():
                
                # Ensure that X & y pointers are arranged sequentially
                sorted_data = sorted(data, key=lambda x: sorted(list(x.tags)))

                curr_worker = self._aliases[worker_id]
                data_ptr = sy.BaseDataset(*sorted_data)
                datasets[curr_worker] = data_ptr

            return datasets

        ###########################
        # Implementation Footnote # 
        ###########################
    
        # TTP should not have residual tensors during training, but will have
        # them during evaluation, as a result of trained models being loaded.
        
        # Retrieve Pointer Tensors to remote datasets
        train_datasets = convert_to_datasets("#train")
        eval_datasets = convert_to_datasets("#evaluate")
        test_datasets = convert_to_datasets("#predict")
        
        return train_datasets, eval_datasets, test_datasets
    

    # def build_reference_plan(
    #     self, 
    #     *sources: Dict[sy.WebsocketClientWorker, sy.BaseDataset],
    # ):
    #     """ Dynamically builds global model reference plan using grid sources

    #     Args:
    #         *sources: Any no. of dataset mappings detected
    #     """
    #     if not self.reference.is_built:

    #         # Ensure that all datasets have the same shape
    #         all_source_shapes = {
    #             (1,) + tuple(dataset.data.shape)[1:]    # arbitrary no. of observations
    #             for meta_datasets in sources
    #             for worker, dataset in meta_datasets.items()
    #         }
    #         assert len(all_source_shapes) == 1
            
    #         reference_shape = max(all_source_shapes)
    #         logging.debug(f"Reference shape: {reference_shape} {type(reference_shape)}")

    #         self.reference.build(shape=reference_shape)


    def convert_to_FL_batches(self, 
        train_datasets: dict, 
        eval_datasets: dict, 
        test_datasets: dict,
        shuffle: bool=True
    ) -> Tuple[sy.FederatedDataLoader, ...]: 
        """ Supplementary function to convert initialised datasets into SGD
            compatible dataloaders in the context of PySyft's federated learning
            
        Args:
            train_datasets (dict(sy.BaseDataset)): 
                Distributed datasets for training
            eval_datasets  (dict(sy.BaseDataset)): 
                Distributed dataset for verifying performance
            test_datasets  (dict(sy.BaseDataset)): 
                Distributed dataset for to be tested on
            shuffle (bool): Toggles the way the minibatches are generated
        Returns:
            train_loader (sy.FederatedDataLoader)
            eval_loader  (sy.FederatedDataLoader)
            test_loader  (sy.FederatedDataLoader)
        """
    
        def construct_FL_loader(dataset, **kwargs):
            """ Cast paired data & labels into configured tensor dataloaders
            Args:
                dataset (list(sy.BaseDataset)): A tuple of X features & y labels
                kwargs: Additional parameters to configure PyTorch's Dataloader
            Returns:
                Configured dataloader (th.utils.data.DataLoader)
            """
            federated_dataset = sy.FederatedDataset(dataset)

            federated_data_loader = sy.FederatedDataLoader(
                federated_dataset, 
                batch_size=(
                    self.arguments.batch_size 
                    if self.arguments.batch_size 
                    else len(federated_dataset)
                ), 
                shuffle=shuffle,
                iter_per_worker=True,   # for LVL 1A parallelization
                #iter_per_worker=False,  # for LVL 1B parallelization
                **kwargs
            )

            return federated_data_loader

        ##############################
        # For LVL 1A parallelization #
        ##############################

        # Load datasets into a configured federated dataloader
        train_loader = construct_FL_loader(train_datasets.values())
        eval_loader = construct_FL_loader(eval_datasets.values())
        test_loader = construct_FL_loader(test_datasets.values())
        
        return train_loader, eval_loader, test_loader

        ##############################
        # For LVL 1B parallelization #
        ##############################

        # train_loaders = {
        #     worker: construct_FL_loader([dataset]) 
        #     for worker, dataset in train_datasets.items()
        # }

        # eval_loaders = {
        #     worker: construct_FL_loader([dataset]) 
        #     for worker, dataset in eval_datasets.items()
        # }

        # test_loaders = {
        #     worker: construct_FL_loader([dataset]) 
        #     for worker, dataset in test_datasets.items()
        # }

        # return train_loaders, eval_loaders, test_loaders


    def load_algorithm(
        self, 
        train_loader: sy.FederatedDataLoader, 
        eval_loader: sy.FederatedDataLoader, 
        test_loader: sy.FederatedDataLoader
    ) -> algorithms.BaseAlgorithm:
        """ Uses specified environment parameters to initialise an algorithm
            object for subsequent use in training and inference.

        Args:
            train_loader (sy.FederatedDataLoader): Training data in configured batches
            eval_loader (sy.FederatedDataLoader): Validation data in configured batches
            test_loader (sy.FederatedDataLoader): Testing data in configured batches
        Returns:
            Algorithm (BaseAlgorithm)
        """ 
        # if not self.reference.is_built:
        #     raise ValueError("Reference plan has not been built!")

        algorithm = getattr(algorithms, self.arguments.algorithm)
        if not algorithm:
            raise AttributeError(f"Specified algorithm '{self.arguments.algorithm}' is not supported!")

        return algorithm(
            action=self.action,
            crypto_provider=self.crypto_provider,
            workers=self.workers,
            arguments=self.arguments,
            train_loader=train_loader,
            eval_loader=eval_loader,
            test_loader=test_loader,
            global_model=self.reference,#.copy(),#copy.deepcopy(self.reference), # copy of reference 
            out_dir=self.out_dir
        )

    ##################
    # Core functions #
    ##################
    
    def load(
        self, 
        shuffle: bool = True, 
        archive: dict = None, 
        version: Tuple[str, str] = None
    ) -> algorithms.BaseAlgorithm:
        """ Prepares federated environment for training or inference. If archive
            is specified, restore all models tracked, otherwise, the default
            global model is used. All remote datasets will be loaded into 
            Federated dataloaders for batch operations if data has not already 
            been loaded. Note that loading is only done once, since searching 
            for datasets within a grid will cause query results to accumulate 
            exponentially on TTP. Hence, this function will only load datasets 
            ONLY if it has NOT already been loaded.

            Note:
                When `shuffle=True`, federated dataloaders will SHUFFLE datasets
                to ensure that proper class representation is covered in each
                minibatch generated. This an important aspect during training.

                When `shuffle=False`, federated dataloaders will NOT shuffle 
                datasets before minibatching. This is to ensure that the
                prediction labels can be re-assembled, aligned and restored on
                the worker nodes during evaluation/inference
            
        Args:
            archive (dict): Paths to exported global & local models
            shuffle (bool): Toggles the way the minibatches are generated
        Returns:
            Initialised algorithm (BaseAlgorithm)
        """
        if not self.is_loaded():
            
            # Extract data pointers from workers
            train_datasets, eval_datasets, test_datasets = self.setup_FL_env()

            # # Build model reference plan(s)
            # self.build_reference_plan(
            #     train_datasets, 
            #     eval_datasets, 
            #     test_datasets
            # )
            
            # Generate federated minibatches via loaders 
            train_loader, eval_loader, test_loader = self.convert_to_FL_batches(
                train_datasets, 
                eval_datasets,
                test_datasets,
                shuffle=shuffle
            )

            # Initialise specified algorithm
            self.algorithm = self.load_algorithm(
                train_loader=train_loader,
                eval_loader=eval_loader,
                test_loader=test_loader
            )

        if archive:
            self.algorithm.restore(archive=archive)

        return self.algorithm

        
    def fit(self) -> Tuple[Model, Dict[str, Model]]:
        """ Performs a remote federated learning cycle leveraging PySyft.

        Returns:
            trained global model (Model)
            Cached local models  (dict(Model))
        """
        if not self.is_loaded():
            raise RuntimeError("Grid data has not been aggregated! Call '.load()' first & try again.")
            
        # Run training using specified algorithm
        self.algorithm.fit()
        
        return self.algorithm.global_model, self.algorithm.local_models


    def evaluate(
        self, 
        metas: List[str] = [], 
        workers: List[str] = [], 
        **kwargs
    ) -> Dict[str, Dict[str, th.Tensor]]:
        """ Using the current instance of the global model, performs inference 
            on pre-specified datasets.

        Args:
            metas (list(str)): Meta tokens indicating which datasets are to be
                evaluated. If empty (default), all meta datasets (i.e. training,
                validation and testing) will be evaluated
            workers (list(str)): Worker IDs of workers whose datasets are to be
                evaluated. If empty (default), evaluate all workers' datasets. 
        Returns:
            Inferences (dict(worker_id, dict(result_type, th.tensor)))
        """
        if not self.is_loaded():
            raise RuntimeError("Grid data has not been aggregated! Call '.load()' first & try again.")

        # Run algorithm for evaluation
        inferences, losses = self.algorithm.evaluate(
            metas=metas,
            workers=workers
        )
        
        return inferences, losses
      
    
    def reset(self):
        """ Original intention was to make this class reusable, but it seems 
            like remote modification of remote datasets is not allowed/does not
            work. Only in the local machine itself do the following functions 
            perform as documented:
            
            1) rm_obj(self, remote_key:Union[str, int])
            2) force_rm_obj(self, remote_key:Union[str, int])
            3) de_register_obj(self, obj:object, _recurse_torch_objs:bool=True)
            
            In hindside, this makes sense since the system becomes more stable. 
            Clients using the grid cannot modify the original datasets in remote 
            workers, mitigating possibly malicious intent. However, this also 
            means that residual tensors will pile up after each round of FL, 
            which will consume more resources. TTP can clear its own objects, 
            but how to inform remote workers to refresh their
            WebsocketServerWorkers?
            
            A possible solution is to leverage on the external Flask 
            interactions.
        """
        raise NotImplementedError
    
    
    def export(self) -> dict:
        """ Exports the global model state dictionary to file
        
        Returns:
            Paths to exported files (dict)
        """
        return self.algorithm.export()