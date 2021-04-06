#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import asyncio
import math
import os
import random
import timeit
from timeit import default_timer as timer
from collections import OrderedDict
from pathlib import Path
from typing import Tuple, List, Dict, Union

# Libs
import numpy as np
import syft as sy
import torch as th
from syft.workers.websocket_client import WebsocketClientWorker

# Custom
import rest_rpc
from rest_rpc.connection.core.utils import RegistrationRecords, RunRecords
from synalgo.arguments import Arguments
from synalgo.early_stopping import EarlyStopping
from synalgo.model import Model
from synalgo.algorithms.base import BaseAlgorithm
# from rest_rpc.evaluation.core.utils import Analyser 

##################
# Configurations #
##################

seed_threshold = 0.15
metric = 'accuracy'
auto_align = True
metas=['train']
combination_keys = {
    'project_id': "fedlearn_project",
    'expt_id': "fedlearn_experiment_1", 
    'run_id': "fedlearn_run_1_1"
}
max_initial_training_epochs = 2
max_initial_pruning_rounds = 10
min_initial_pruning_size = 0.3
reconfig_interval = 2
T_CONSTANT = 0.05

amundsen_metadata = {}

########################################
# Federated Algorithm Class - FedPrune #
########################################

class FedPrune(BaseAlgorithm):
    """ 
    Implements the FedPrune algorithm.

    NOTES:
    1)  Any model (i.e. self.global_model or self.local_model[w_id]) passed in 
        has an attribute called `layers`.
        - `layers` is an ordered dict
        - All nn.module layers are set as attributes in the Model object
            eg. 
            
            A model of this structure:
            [
                {
                    "activation": "relu",
                    "is_input": True,
                    "l_type": "Conv2d",
                    "structure": {
                        "in_channels": 1, 
                        "out_channels": 4, # [N, 4, 28, 28]
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1
                    }
                },
                {
                    "activation": None,
                    "is_input": False,
                    "l_type": "Flatten",
                    "structure": {}
                },
                # ------------------------------
                {
                    "activation": "softmax",
                    "is_input": False,
                    "l_type": "Linear",
                    "structure": {
                        "bias": True,
                        "in_features": 4 * 28 * 28,
                        "out_features": 1
                    }
                }

            ]
            will have:
            model.nnl_0_conv2d  --> Conv2d( ... )
            model.nnl_1_flatten --> Flatten()
            model.nnl_2_linear  --> Linear( ... )
        - Structure of `layers`:
            {
                layer_1_name: activation_function_1,
                layer_2_name: activation_function_2,
                ...
            }
        - If a layer is not paired with an activation function, a default
          identity function will be registered.
            {
                layer_1_name: activation_function_1,
                layer_2_name: lambda x: x,              <---
                ...
            }

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
        # next time will have kwargs with federated combination
    ):
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

        ### Existing Attributes (for your reference) ###
        self.action = action
        # self.crypto_provider = crypto_provider
        self.workers = workers
        print("Arguments:", arguments)
        self.arguments = arguments
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.global_model = global_model  
        self.local_models = local_models
        # self.loss_history = {
        #     'global': {
        #         'train': {},
        #         'evaluate': {}
        #     },
        #     'local': {}
        # }
        # self.loop = None
        # self.out_dir = out_dir
        # self.checkpoints = {}


        # General attributes
        self.selected_worker = None

        # Network attributes


        # Data attributes
        self.per_layer_input_sizes = None
        self.layer_timings = None
        self.masks = None
        
        # Model attributes


        # Optimisation attributes


        # Export Attributes



    ###########
    # Helpers #
    ###########

    def get_fraction_of_non_zero_prunable(self, round_number):
        return 0.3 * math.pow(0.5, round_number/10000)


    def find_layer(self, pos):
        for idx, c in enumerate(self.cumulative_n_weights):
            if pos < c:
                return idx
        raise Exception


    def get_cumulative_n_weights(self):
        layers = list(self.get_layers_with_weights(self.global_model).values())
        cumulative_n_weights = [0] * len(layers)
        for i in range(len(layers)):
            if i == 0:
                cumulative_n_weights[0] = th.numel(layers[i].weight)
            else:
                cumulative_n_weights[i] = cumulative_n_weights[i-1] + th.numel(layers[i].weight)
        print('cumulative_n_weights', cumulative_n_weights)
        self.cumulative_n_weights = cumulative_n_weights


    def parse_layers(self) -> dict:
        """
        Returns:
            Name-to-Layer mappings (dict)
        """
        return {
            layer_name: getattr(self.global_model, layer_name)
            for layer_name, _ in self.global_model.layers.items()
        }   


    def get_layers_with_weights(self, model):
        """
        Args:
            model (Model)
        Returns:
            Name-to-Layers with weights mappings (dict)
        """
        d = OrderedDict()
        for layer_name, layer in model.layers.items():
            if hasattr(getattr(model, layer_name) , 'weight'):
                d[layer_name] = getattr(model, layer_name)        
        return d


    def initialise_layer_inputs(self, model, dataset):
        """
        < Objective - Parse the model to get a list of tensors whose shapes
            correspond to each layer of the specified model.

        Args:

        Returns:
            Tensor dict mapping layer names to the input tensors that 
            correspond to these layers
        """
        name_to_layer_mapping = self.parse_layers()

        # < Insert for loop over here >
        per_layer_input_sizes = {}

        done = False
        for batch in dataset:
            if not done:
                for worker, (data, labels) in batch.items():
                    if worker is self.selected_worker:
                        for idx, (layer_name, layer) in enumerate(name_to_layer_mapping.items()):
                            # < Given the layer, generate its respective input tensor >   
                            previous_layers = list(name_to_layer_mapping.values())[0:idx]
                            submodel = th.nn.Sequential(*previous_layers)
                            input_tensor = None
                            print("Submodel:", submodel)
                            if hasattr(layer, 'weight'):
                                t = submodel.send(worker)
                                t.train()
                                outputs = t(data)
                                t.get()
                                print("outputs shape,", layer_name, outputs.shape)
                                input_tensor = th.randn(outputs.shape, dtype=data.dtype)
                                per_layer_input_sizes[layer_name] = input_tensor
                    done = True
                    break

        self.per_layer_input_sizes = per_layer_input_sizes
        return self.per_layer_input_sizes


    def select_optimal_worker(self):
        """ Choose a worker for initial training that can help us obtain a
            seeding model that is most generalised

        < For now use random, but need to implement the entropy measure >
        """
        self.selected_worker = random.choice(self.workers)
        self.selected_worker = self.workers[0]
        return self.selected_worker


    def measure_layer_times(self, selected_worker, datasets, criterion, optimizer) -> Dict[str, float]:
        """ 
        < Objective - Obtain measurements for all layers

        How to use:
        selected_worker = random.choice([self.workers])
        layer_timings = self.measure_layer_times(selected_worker)

        Args:
            worker (WebsocketClientWorker)
        """
        name_to_layer_mapping = self.parse_layers()
        for k, v in self.per_layer_input_sizes.items():
            print(k, v.shape)
        layer_timings = OrderedDict()
        for layer_name, input_tensor in self.per_layer_input_sizes.items():
            layer = name_to_layer_mapping[layer_name]

            submodel = th.nn.Sequential(layer)
            layer_time = 0.1
            
            def time_single_layer(submodel, input_tensor):
                # Send input tensor to specified worker to convert to PointerTensor
                for batch in datasets:
                    for worker, (data, labels) in batch.items():
                        if worker is self.selected_worker:
                            print('data', data.shape, data.dtype)
                            print('labels', labels.shape, labels.dtype)
                            t = submodel.send(worker)
                            s = input_tensor.send(worker)
                            print("inputs shape, labels shape", input_tensor.shape, labels.shape)

                            t.train()
                            # optimizer.zero_grad() 
                            # Forward Propagation
                            o = t(s)
                            # o = th.ones_like(o).long() # long() otherwise 'one_hot is only applicable to index tensor'
                            # but also RuntimeError: "log_softmax_lastdim_kernel_impl" not implemented for 'Long'
                            print("o", o.shape, o.shape[0])
                            flabels = th.randn(o.shape[0])
                            print('flabels', flabels.shape)
                            flabelss = flabels.send(worker) # RuntimeError: expected scalar type Long but found Float
                            loss = criterion(
                                outputs=o, 
                                labels=flabelss,
                                w=self.global_model.state_dict(),
                                wt=self.global_model.state_dict()
                            ) 
                            # With BCEWithLogitsLoss,  in base.py forward(), get Formatted labels type: torch.Size([8, 300, 300]) torch.FloatTensor
                            # ValueError: Target size (torch.Size([8, 300, 300])) must be the same as input size (torch.Size([8, 300]))
                            # Backward propagation
                            loss.backward()
                            #curr_global_model = curr_global_model.get()
                            submodel = t.get()
            layer_time = timeit.timeit("time_single_layer(submodel, input_tensor)", globals=locals())
            # time_single_layer(submodel,input_tensor)
            layer_timings[layer_name] = layer_time            

        self.layer_timings = layer_timings
        print(self.layer_timings)
        return self.layer_timings


    def count_model_parameters(self) -> int:
        """ Counts the total number of parameters in the current global model

        Returns:
            Total parameter count (int)
        """
        layers_with_weights = self.get_layers_with_weights(self.global_model)
        return sum([
            th.numel(layer.weight) 
            for _, layer in layers_with_weights.items()
        ])


    def generate_mask(self) -> List[int]:
        """ Generates mask for phase 1 & phase 2 operations

        Returns:
            Mask (list(int))
        """
        num_weight_elements = self.count_model_parameters()
        mask = [1] * num_weight_elements
        return mask


    def perform_pruning(
        self,
        clients_importances: List,
        epoch: int
    ):
        """ Performs the pruning algorithm, updating self.masks

        Args:
            clients_importances  (List): Collected importance measures (sum of squared gradients) from workers
            epoch (int): Current epoch of training
        Returns:
            trained local models
        """ 
        with th.no_grad():
            layers = self.get_layers_with_weights(self.global_model)
            flat_mask = th.cat([th.flatten(m) for m in self.masks.values()])
            flattened_weights = th.cat([th.flatten(layer.weight) for _, layer in layers.items()])
            
            flat_mask_reapplication_tensor = th.tensor(flat_mask)
            weights_with_mask_applied = flattened_weights * flat_mask_reapplication_tensor
            unmasked_weights = weights_with_mask_applied[th.nonzero(weights_with_mask_applied, as_tuple=True)].tolist() # as_tuple allows this indexing

            print("flattened weights:", len(flattened_weights), 'unmasked_weights', len(unmasked_weights), '0s in previous round mask', len(flat_mask) - len(th.nonzero(flat_mask_reapplication_tensor)), 'fraction masked', (len(flat_mask) - len(th.nonzero(flat_mask_reapplication_tensor))) / len(flat_mask))
            sorted_unmasked_weights = sorted(unmasked_weights, key=abs) # sort from smallest magnitude weight to largest
            fraction_unmasked_prunable = self.get_fraction_of_non_zero_prunable(epoch)
            n_unmasked_prunable = int(fraction_unmasked_prunable * len(unmasked_weights)) # n smallest prunable weights
            print("n_unmasked_prunable", n_unmasked_prunable, "fraction_unmasked_prunable", fraction_unmasked_prunable)
            print('first and last elements of sorted_unmasked_weights', sorted_unmasked_weights[0], sorted_unmasked_weights[-1])
            
            largest_prunable_threshold = abs(sorted_unmasked_weights[n_unmasked_prunable]) # add elements greater than threshold to P bar
            print('largest_prunable_threshold', largest_prunable_threshold)

            set_P_bar = set()
            flattened_weights = flattened_weights.tolist()
            
            for idx, w in enumerate(flattened_weights):
                if flat_mask[idx] == 1 and abs(w) >= largest_prunable_threshold:
                    set_P_bar.add(idx)
            

            flat_mask = [0] * self.num_weight_elements
            for item in set_P_bar:
                flat_mask[item] = 1

            averaged_ssgs = [None for x in range(len(layers))] # Take the average SSGs across clients
            averaged_ssgs_divided_by_time = [None for x in range(len(layers))]
            for idx in range(len(layers)):
                averaged_ssgs[idx] = th.sum(th.stack([ci[idx] for ci in clients_importances], dim=0), dim=0) / len(clients_importances) 
                averaged_ssgs_divided_by_time[idx] = th.sum(th.stack([ci[idx] for ci in clients_importances], dim=0), dim=0) / len(clients_importances)  / list(self.layer_timings.values())[idx]
            flattened_ssgs = th.cat([th.flatten(ssg) for ssg in averaged_ssgs]).tolist()
            flattened_asdt = th.cat([th.flatten(asdt) for asdt in averaged_ssgs_divided_by_time])
            flattened_asdt.numpy()
    
            if len(flat_mask) != sum([th.numel(s) for s in averaged_ssgs]):
                raise Exception

            sort_start = timer()
            argsorted_flattened_asdt = np.argsort(-flattened_asdt).tolist() # asdt (avg squared grads divided by time) is gsquared / t. sort in descending order and represent as list instead of Tensor

            algo_start = timer()
            S = [x for x in argsorted_flattened_asdt if x not in set_P_bar]  # remove indices that exist in P_bar
            print("Size of all elements:", self.num_weight_elements, "size of p bar:", len(set_P_bar), 'fraction in p bar:', len(set_P_bar)/self.num_weight_elements, "S:", len(S))
            def get_time(pos): # for a single position
                return list(self.layer_timings.values())[self.find_layer(pos)]

            def delta(set_of_pos): # a set
                squared_params = [flattened_ssgs[x] for x in set_of_pos]
                return sum(squared_params)


            # T_CONSTANT is 'c' in T(M) = c + sum(t of j)
            def T(set_of_pos, constant=T_CONSTANT):
                ret = constant + sum([get_time(i) for i in set_of_pos])
                return ret

            def gamma(set_of_pos):
                return delta(set_of_pos)/T(set_of_pos)

            set_A = []

            current_delta = delta(set_P_bar)
            current_T = T(set_P_bar) # 5 seconds for this?
            current_gamma = 0 if current_delta == 0 else current_delta / current_T
            print("current_gamma, delta, T:", current_gamma,current_delta, current_T )
            for j in range(len(S)):
                index = S[j] # original index of this jth element 
                gsquared_t_j = flattened_asdt[index]

                if gsquared_t_j >= current_gamma:
                    set_A.append(index)
                    current_delta = current_delta + flattened_ssgs[index]
                    current_T = current_T + get_time(index)
                    current_gamma = current_delta / current_T
                else: 
                    break
            print("new gamma, delta, T", current_gamma, current_delta, current_T)
            for item in set_A:
                flat_mask[item] = 1
            flat_mask_tensor = th.tensor(flat_mask)
            fraction_remaining =  len(th.nonzero(flat_mask_tensor))/len(flat_mask_tensor)
            print("Epoch", epoch, ": items added to set A", len(set_A), "1s in mask:" ,len(th.nonzero(flat_mask_tensor)), "zeros in mask", len(flat_mask)-len(th.nonzero(flat_mask_tensor)), "fraction pruned", 1-len(th.nonzero(flat_mask_tensor))/len(flat_mask_tensor), "fraction remaining", round(fraction_remaining,5)) 
            
            splits = th.split(flat_mask_tensor,[th.numel(layer.weight) for layer in layers.values()], dim=0)
            reshaped = [splits[idx].reshape(layer.weight.shape) for idx, layer in enumerate(layers.values())]
            for mask in reshaped:
                elements_remaining = len(th.nonzero(mask))
                mask_length = len(mask.flatten())
                remaining_density = round(elements_remaining/mask_length, 3)
                print("reshaped:", elements_remaining, mask_length, remaining_density)
            
            layer_idx = 0
            for k in self.masks: 
                self.masks[k] = reshaped[layer_idx]
                layer_idx += 1


    def perform_initial_training(
        self,
        selected_worker: WebsocketClientWorker,
        datasets: dict,
        models: dict,
        optimizer: th.optim, 
        scheduler: th.nn, 
        criterion: th.nn, 
        stopper: EarlyStopping, 
        metric: str,
        max_epochs: int = 10
    ):
        """ Phase 1 dictates initial training to seed the model for better 
            convergence in Phase 2

        """
        curr_global_model =  self.global_model
        
        for epoch in range(max_epochs):

            # Train global model through the entire batch once
            for batch in datasets:
                for worker, (data, labels) in batch.items():
                    if worker == selected_worker:
                        
                        curr_global_model = curr_global_model.send(worker)

                        # Zero gradients to prevent accumulation  
                        curr_global_model.train()
                        curr_global_model.zero_grad()

                        # Forward Propagation
                        outputs = curr_global_model(data)

                        loss = criterion(
                            outputs=outputs, 
                            labels=labels,
                            w=curr_global_model.state_dict(),
                            wt=curr_global_model.state_dict()
                        )

                        # Backward propagation
                        loss.backward()
                        optimizer.step()

                        curr_global_model = curr_global_model.get()

            # Use trained initialised global model for inference
            worker_inferences, _ = self.perform_FL_evaluation(
                datasets=datasets,
                workers=[selected_worker],
                is_shared=False
            )
            # Convert collection of object IDs accumulated from minibatch 
            analyser = rest_rpc.evaluation.core.utils.Analyser(
                inferences=worker_inferences, 
                metas=metas,
                **combination_keys
            )
            # '''
            try:
                reg_records = RegistrationRecords()
                all_records = reg_records.read_all()
                print('all records', all_records)
                # Sorted inferences: []
                print("worker inferences", worker_inferences)
                polled_stats = analyser.infer(reg_records=all_records) # Retrieve scores from entire grid
                print(polled_stats) # {}
                fl_combination = tuple(combination_keys.keys())
                metric_value = polled_stats[fl_combination][selected_worker][metric] 
                print(metric_value)
                if metric_value > seed_threshold:
                    break
            except Exception as e:
                print("Exception")
                print(e)
                pass
            # '''
            
        return self.global_model

    def perform_initial_pruning(
        self,
        selected_worker: WebsocketClientWorker,
        datasets: dict,
        models: dict,
        optimizer: th.optim, 
        scheduler: th.nn, 
        criterion: th.nn, 
        stopper: EarlyStopping, 
        metric: str,
        max_epochs: int = 10,
        min_model_fraction: float = 0.2,
        reconfig_interval: int = 2
    ):
        """ Phase 1 dictates initial training to seed the model for better 
            convergence in Phase 2

            {
                "train": {
                    "y_pred": [
                        [0.],
                        [1.],
                        [0.],
                        [1.],
                        .
                        .
                        .
                    ],
                    "y_score": [
                        [0.4561681891162],
                        [0.8616516118919],
                        [0.3218971919191],
                        [0.6919811999489],
                        .
                        .
                        .
                    ]
                },
                "evaluate": {},
                "predict": {}
            }
        """
        curr_global_model =  self.global_model
        layers = self.get_layers_with_weights(curr_global_model)
        
        model_sizes = [] # keep track of the model size to check if it is stable over 5 consecutive reconfigurations

        masks = OrderedDict()
        # initialise masks of ones
        for name, layer in layers.items():
            masks[name] = th.ones_like(layer.weight)
        self.masks = masks
        workers_ssgs = {}
        
        local_masks = {}
        for worker in self.workers:
            local_masks[worker] = {name: mask.clone().detach().send(worker) for name, mask in self.masks.items()}
            
        for epoch in range(max_epochs):
            print("Initial pruning: Epoch", epoch)
            for batch in datasets:
                for worker, (data, labels) in batch.items():
                    if worker == selected_worker:
                        curr_global_model = curr_global_model.send(worker)
                        # curr_local_model = curr_local_model.send(worker)
                        # Zero gradients to prevent accumulation  
                        curr_global_model.train()
                        curr_global_model.zero_grad()

                        # Mask weights
                        local_mask = local_masks[worker] 
                        original_weights = {}
                        for layer_name, mask in local_mask.items():
                            layer = getattr(curr_global_model, layer_name)
                            original_weights[layer_name] =  layer.weight.clone().detach()
                            layer.weight = th.nn.Parameter(th.mul(layer.weight, mask))

                        # Forward Propagation
                        outputs = curr_global_model(data)
                        loss = criterion(
                            outputs=outputs, 
                            labels=labels,
                            w=curr_global_model.state_dict(),
                            wt=curr_global_model.state_dict()
                        )
                        # Backward propagation
                        loss.backward()
                        with th.no_grad():
                            if workers_ssgs.get(worker.id) is None:
                                workers_ssgs[worker.id] = [None for x in range(len(layers))] 
                            
                            for idx, layer_name in enumerate(local_mask):
                                layer = getattr(curr_global_model, layer_name)
                                summed_squared_gradients = workers_ssgs[worker.id]
                                if summed_squared_gradients[idx] is None:
                                    summed_squared_gradients[idx] = layer.weight.grad.clone().detach() * layer.weight.grad.clone().detach()
                                else:
                                    summed_squared_gradients[idx] = summed_squared_gradients[idx] + (layer.weight.grad.clone().detach() * layer.weight.grad.clone().detach()) 
                            # After masking the weights above, when doing * or +:  clone() is required otherwise
                            # RuntimeError: Command "__mul__"  is not a supported torch operation  (or "__add__" )
                        # Mask the gradients before optimiser updates the gradients
                        for layer_name, mask in local_mask.items():
                            layer = getattr(curr_global_model, layer_name)
                            layer.weight.grad = layer.weight.grad.clone().detach() * mask
                        optimizer.step()
                        
                        # Restore weights that were masked to their original values
                        with th.no_grad():
                            for layer_name, mask in local_mask.items():
                                layer = getattr(curr_global_model, layer_name)
                                new_weight =  layer.weight * mask + (1-mask) * original_weights[layer_name]
                                layer.weight = th.nn.Parameter(new_weight)
                            
                        curr_global_model = curr_global_model.get()
                        self.global_model = curr_global_model

            if epoch != 0 and epoch % reconfig_interval == 0:
                # Perform reconfiguration
                print("    Reconfig round at epoch:", epoch)

                # Collect remote gradients 
                clients_importances = []
                for k, v in workers_ssgs.items():
                    clients_importances.append([x.get() for x in v])
                workers_ssgs = {}   # Reset dict

                self.perform_pruning(clients_importances, epoch)
                
                flat_mask = th.cat([th.flatten(m) for m in self.masks.values()])
                model_size = len(th.nonzero(th.tensor(flat_mask)))
                model_sizes.append(model_size)

                if model_size/self.num_weight_elements < min_initial_pruning_size:
                    print("model size small", model_size)
                    break
                is_stable = False
                if len(model_sizes) >= 5:
                    is_stable = True
                    for i in range(-4,0):
                        relative_change = abs((model_sizes[i] - model_sizes[i-1]) / model_sizes[i])
                        if (relative_change >= 0.10):
                            is_stable = False
                if is_stable:
                    print("model is stable", model_sizes )
                    break

        return self.global_model

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
        epochs: int,
        workers_ssgs: dict,
        local_masks: dict
    ):
        """ Parallelizes training across each distributed dataset 
            (i.e. simulated worker) Parallelization here refers to the 
            training of all distributed models per epoch.
            Note: All objects involved in this set of operations have
                already been distributed to their respective workers

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
            workers_ssgs (dict): 
            local_masks (dict): 
        Returns:
            trained local models
        """ 
        # Tracks which workers have reach an optimal/stagnated model
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
            print("---- train worker start", worker, worker.id)

            # Extract essentials for training
            curr_global_model = cache[worker]
            curr_local_model = models[worker]
            curr_optimizer = optimizers[worker]
            curr_criterion = criterions[worker]
            summed_squared_gradients = workers_ssgs[worker]
            local_mask = local_masks[worker]

            layers = self.get_layers_with_weights(curr_local_model)
             
            # Check if worker has been stopped
            if worker.id not in WORKERS_STOPPED:

                curr_global_model = curr_global_model.send(worker)
                curr_local_model = curr_local_model.send(worker)

                # Zero gradients to prevent accumulation  
                curr_local_model.train()
                curr_optimizer.zero_grad() 

                # Mask weights
                original_weights = {}
                for layer_name, mask in local_mask.items():
                    layer = getattr(curr_local_model, layer_name)
                    original_weights[layer_name] =  layer.weight.clone().detach()
                    layer.weight = th.nn.Parameter(th.mul(layer.weight, mask))

                # Forward Propagation
                outputs = curr_local_model(data)
                loss = curr_criterion(
                    outputs=outputs, 
                    labels=labels,
                    w=curr_local_model.state_dict(),
                    wt=curr_global_model.state_dict()
                )
                # Backward propagation
                loss.backward()
                with th.no_grad():
                    for idx, layer_name in enumerate(local_mask):
                        layer = getattr(curr_local_model, layer_name)
                        if summed_squared_gradients[idx] is None:
                            summed_squared_gradients[idx] = layer.weight.grad.clone().detach() * layer.weight.grad.clone().detach()
                        else:
                            summed_squared_gradients[idx] = summed_squared_gradients[idx] + (layer.weight.grad.clone().detach() * layer.weight.grad.clone().detach()) 
                # Mask the gradients before optimiser updates the gradients
                for layer_name, mask in local_mask.items():
                    layer = getattr(curr_local_model, layer_name)
                    layer.weight.grad = layer.weight.grad.clone().detach() * mask
                curr_optimizer.step()
                
                # Restore masked weights
                with th.no_grad():
                    for layer_name, mask in local_mask.items():
                        layer = getattr(curr_local_model, layer_name)
                        new_weight =  layer.weight * mask + (1-mask) * original_weights[layer_name]
                        layer.weight = th.nn.Parameter(new_weight)
                
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
                worker (WebsocketClientWorker): Worker to be evaluated
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
                    curr_scheduler.step()

            assert schedulers[worker] is curr_scheduler
            assert stoppers[worker] is curr_stopper 

        async def train_datasets(datasets):
            """ Train all batches in a composite federated dataset """
            # Note: All TRAINING must be synchronous w.r.t. each batch, so
            #       that weights can be updated sequentially!
            for batch in datasets:
                await train_batch(batch)
                            
            # stagnation_futures = [
            #     check_for_stagnation(worker) 
            #     for worker in self.workers
            # ]
            # await asyncio.gather(*stagnation_futures)


        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        print("&&&&&&&&&&&&&&&&&&& Perform parallel training")

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


    def finalize_mask(self):
        """ Reconstruct pruned global model according to the finalised mask

        < Modifies the global model inplace >
        """
        pass

    ##################
    # Core functions #
    ##################

    def fit(self):
        """ Performs federated training using a pre-specified model as
            a template, across initialised worker nodes, coordinated by
            a ttp node.
            
        Returns:
            Trained global model (Model)
        """

        optimizer = self.arguments.optimizer( 
            **self.arguments.optimizer_params,
            params=self.global_model.parameters()
        )
        # scheduler = self.arguments.lr_scheduler(
        #     **self.arguments.lr_decay_params,
        #     optimizer=optimizer
        # )
        criterion = self.build_custom_criterion()(
            **self.arguments.criterion_params
        )
        self.num_weight_elements = self.count_model_parameters()
        self.select_optimal_worker()
        self.get_cumulative_n_weights()
        self.initialise_layer_inputs(self.global_model, self.train_loader)

        self.measure_layer_times(
            self.selected_worker,
            self.train_loader,
            criterion,
            optimizer
        )

        try:
            import sys, traceback, logging
            # logging.basicConfig(level=logging.DEBUG)

            # Phase 0
            self.perform_initial_training(
                selected_worker=self.selected_worker,
                datasets=self.train_loader,
                models=None,
                optimizer=optimizer,
                scheduler=None,
                criterion=criterion,
                stopper=None,
                metric=metric,
                max_epochs=max_initial_training_epochs
            )

            # Phase 1: Perform train + collect gradients + pruning on single worker
            self.perform_initial_pruning(
                selected_worker=self.selected_worker,
                datasets=self.train_loader,
                models=None,
                optimizer=optimizer,
                scheduler=None,
                criterion=criterion,
                stopper=None,
                metric=metric,
                max_epochs=max_initial_training_epochs
            )

            # Phase 2 Operation: Federated masked averaging + Pruning
            current_round = 0
            while current_round < self.arguments.rounds:
                # Copy the initial-pruned global model
                local_models = self.generate_local_models()
                prev_models = self.generate_local_models()

                optimizers = {
                    w: self.arguments.optimizer( 
                        **self.arguments.optimizer_params,
                        params=model.parameters()
                    ) for w, model in local_models.items()
                }
                criterions = {
                    w: self.build_custom_criterion()(
                        **self.arguments.criterion_params
                    ) for w,m in local_models.items()
                }
                # Perform train + collect gradients + pruning on all workers
                layers = self.get_layers_with_weights(self.global_model)
                workers_ssgs = {
                    worker: [None for x in range(len(layers))] for worker in self.workers
                }
                local_masks = {}
                for worker in self.workers:
                    local_masks[worker] = {name: mask.clone().detach().send(worker) for name, mask in self.masks.items()}

                (retrieved_models, _, _, _, _) = self.perform_parallel_training(
                    datasets=self.train_loader, 
                    models=local_models,
                    cache=prev_models,
                    optimizers=optimizers, 
                    schedulers=None,
                    criterions=criterions, 
                    stoppers=None,
                    rounds=current_round,
                    epochs=self.arguments.epochs,
                    workers_ssgs=workers_ssgs,
                    local_masks=local_masks
                )

                # Retrieve all models from their respective workers
                logging.debug(f"Current global model:\n {self.global_model.state_dict()}")
                aggregated_params = self.calculate_global_params(
                    self.global_model, 
                    retrieved_models, 
                    self.train_loader
                )

                # Update weights with aggregated parameters 
                self.global_model.load_state_dict(aggregated_params)

                # Collect importance measures
                clients_importances = []
                for k, v in workers_ssgs.items():
                    clients_importances.append([x.get() for x in v])
                epochs_so_far = self.arguments.epochs * (current_round + 1)
                current_round += 1
                
                self.perform_pruning(clients_importances, epochs_so_far)

            self.finalize_mask()

        except Exception as e:
            print("Exception")
            print(e)
            traceback.print_exc()

        finally:
            print("Finally")
            return self.global_model, self.local_models


    def analyse(self):
        """ Calculates contributions of all workers towards the final global 
            model. 
        """
        raise NotImplementedError
