#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import inspect
import asyncio
from collections import OrderedDict
import logging
import copy
from pathlib import Path
from typing import Tuple, List, Dict, Union
import os

# Libs
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker
from tqdm import tqdm

# Custom
from synalgo.arguments import Arguments
from synalgo.early_stopping import EarlyStopping
from synalgo.model import Model
from synalgo.algorithms.base import BaseAlgorithm

##################
# Configurations #
##################

### Specify any new parameters required over here as constants first! ###

### Global states to keep track of all the worker training information
server_logits_dict = {}
extracted_feature_dict = {}
logits_dict = {}
labels_dict = {}

# e.g. model_structure to defined from Synergos Notebook/Driver
# model_structure = [
#     {
#         "activation": None,
#         "is_input": True,
#         "l_type": "Conv2d",
#         "structure": {
#             "in_channels": 1,
#             "out_channels": 32, # [N, 32, 28, 28]
#             "kernel_size": 3,
#             "stride": 1,
#             "padding": 1
#         }
#     },
#     {
#         "activation": None, 
#         "is_input": False,
#         "l_type": "ReLU",
#         "structure": {}
#     },
#     {
#         "activation": None, 
#         "is_input": False,
#         "l_type": "BatchNorm2d",
#         "structure": {
#             "num_features": 32
#         }
#     },
#     {
#         "activation": None,
#         "is_input": False,
#         "l_type": "Conv2d",
#         "structure": {
#             "in_channels": 32,
#             "out_channels": 64, # [N, 64, 28, 28]
#             "kernel_size": 3,
#             "stride": 1,
#             "padding": 1
#         }      
#     },
#     {
#         "activation": None, 
#         "is_input": False,
#         "l_type": "ReLU",
#         "structure": {}
#     },
#     {
#         "activation": None, 
#         "is_input": False,
#         "l_type": "BatchNorm2d",
#         "structure": {
#             "num_features": 64
#         }
#     },
#     {
#         "activation": None,
#         "is_input": False,
#         "l_type": "Conv2d",
#         "structure": {
#             "in_channels": 64,
#             "out_channels": 32, # [N, 32, 28, 28]
#             "kernel_size": 3,
#             "stride": 1,
#             "padding": 1
#         }      
#     },
#     {
#         "activation": None, 
#         "is_input": False,
#         "l_type": "ReLU",
#         "structure": {}
#     },
#     {
#         "activation": None, 
#         "is_input": False,
#         "l_type": "BatchNorm2d",
#         "structure": {
#             "num_features": 32
#         }
#     },

#     {
#         "activation": None,
#         "is_input": False,
#         "l_type": "Flatten",
#         "structure": {}
#     },
#     {
#         "activation": None,
#         "is_input": False,
#         "l_type": "Linear",
#         "structure": {
#             "bias": True,
#             "in_features": 32 * 28 * 28,
#             "out_features": 3
#         }
#     }

# ]

########################################
# Federated Loss Helper Class - KLLoss #
########################################
  
class KLLoss(nn.Module):
    """
    KL Divergence loss function using softmax temperature function

    Args:
        temperature: the amount of information to distill (higher temp -> softer logits)
    """
    def __init__(self, temperature=1):
        super(KLLoss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        return loss



##############################################
# Federated Model Helper Class - WorkerModel #
##############################################

class WorkerModel(nn.Module):
    """
    Generate a worker model using a global model structure
    Cut up till L layers from the global model

    Args:
        L: Slice up till L layers (index by 0)
        structure: The global model structure in OrderedDict format
        shape: The shape of the data in order to determine 
               the in_features of the final output nn.Linear() layer.
    """
    def __init__(self, L, structure, shape):
        super(WorkerModel, self).__init__()
        
        # list for storing the worker layers
        list_layers = []
        
        # slice the model up till L layers
        for i, (layer_name, layer) in enumerate(list(structure.items())[:L]):
            list_layers.append((str(layer_name), layer))

        params = OrderedDict(list_layers)
        self.client_feature_extractor = nn.Sequential(params)
        
        # Get the shape of the data by running through client_feature_extractor
        # in order to determine the no. of in_features for the final output layer
        client_output_in_features = nn.Flatten()(self.client_feature_extractor(th.ones(shape))).shape[1]
        logging.debug(f'worker_model: {structure.items()}')
        
        # check if the final output layer from the global model 
        # is a nn.Linear() layer with the attribute 'out_features'
        if hasattr(list(structure.items())[-1][1], 'out_features'):
           client_output_out_features = list(structure.items())[-1][1].out_features
        else:
            # if softmax function is automatically applied on the final output layer then 
            # final output layer is the second last layer with the attribute 'out_features'
            client_output_out_features = list(structure.items())[-2][1].out_features

        logging.debug(f'client_output_in_features: {client_output_in_features}')
        self.client_output = nn.Linear(client_output_in_features, client_output_out_features)
        
    def forward(self, x):
        x = self.client_feature_extractor(x)
        extracted_feature = x
        logging.debug(f"extracted_feature_shape: {extracted_feature.shape}")

        # Some websocket error when applying
        # nn.Flatten() on a nn.Linear() layer 
        # which resulted in the following exception

        # syft.exceptions.ObjectNotFoundError: Object "66432032110" not found on worker! 
        # You just tried to interact with an object ID: 66432032110 on 
        # <syft.generic.object_storage.ObjectStore object at 0x7fe88482c190> 
        # which does not exist!!!Use .send() and .get() on all your tensors to 
        # make sure they're on the same machines. If you think this tensor does exist, 
        # check the object_store._objects dict on the worker and see for yourself. 
        # The most common reason this error happens is because someone calls .get() 
        # on the object's pointer without realizing it (which deletes the remote 
        # object and sends it to the pointer). Check your code to make sure you 
        # haven't already called .get() on this pointer!

        # use view to flatten the input instead of nn.Flatten().. fixes the exception error.
        x = x.view(x.shape[0], -1)
        x = self.client_output(x)        
        logging.debug(f"output_shape: {x.shape}")
        return x, extracted_feature
            


##############################################
# Federated Model Helper Class - ServerModel #
##############################################

class ServerModel(nn.Module):
    """
    Generate a Server model using a global model structure
    Slice the global model from L layers and onwards

    Args:
        L: Slice from the L layer and onwards
        structure: The global model structure in OrderedDict format
    """
    def __init__(self, L, structure):
        super(ServerModel, self).__init__()        
        
        # storing only the server layers
        list_layers_server = []

        # slice from L layer onwards
        logging.debug(f"ServerModel: {structure}")
        logging.debug(f"ServerModelItems: {list(structure.items())[L:-1]}")

        # During alignment phase, softmax layer is automatically added.
        # Hence we sliced from up till the last layer only.
        server_layers = list(structure.items())[L:-1]
        for _, (layer_name, layer) in enumerate(server_layers):
            list_layers_server.append((str(layer_name), layer))

        params_server = OrderedDict(list_layers_server)
        logging.debug(f"params_server: {params_server}")
        self.server_feature_extractor = nn.Sequential(params_server)
       
    def forward(self, x):
        x = self.server_feature_extractor(x)
        return x



######################################
# Federated Algorithm Class - FedGKT #
######################################

class FedGKT(BaseAlgorithm):
    """ 
    Implements the FedGKT algorithm.

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
        # L: int = 3,
        # temperature: int = 3,
        # alpha: int = 1,
        # whether_distill_on_the_server: bool = True,
        # server_epochs = 5
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
            out_dir=out_dir,
            # L=L,
            # temperature=temperature,
            # alpha=alpha,
            # whether_distill_on_the_server=whether_distill_on_the_server,
            # shape=shape,
            # server_epochs=server_epochs
        )

        # General attributes
        self.action = action

        # FedGKT atrributes
        self.L = arguments.L
        self.temperature = arguments.temperature
        self.alpha = arguments.alpha
        self.whether_distill_on_the_server = arguments.whether_distill_on_the_server
        self.server_epochs = arguments.server_epochs
        
        self.criterion_kl = KLLoss(arguments.temperature)

        # initialize all workers states to empty
        for worker in workers: # workers list
            extracted_feature_dict[worker.id] = {}
            logits_dict[worker.id] = {}
            labels_dict[worker.id] = {}
          
        # Network attributes
        # e.g. server IP and/or port number
        self.crypto_provider = crypto_provider
        self.workers = workers

        # Model attributes 
        self.global_model = global_model # only used as a reference for slicing
        self.server_model = None
        self.local_models = local_models
        self.global_train_loss = None
        self.loss_history = {
            'global': {
                'train': {},
                'evaluate': {}
            },
            'local': {}
        }

        # Data attributes
        # e.g participant_id/run_id in specific format
        self.arguments = arguments
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader

        # Optimisation attributes
        # e.g multiprocess/asyncio if necessary for optimisation


        # Export Attributes 
        # e.g. any artifacts that are going to be exported eg Records


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
                suppport FedProx 
            
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

                logging.debug(f"Action: {ACTION}, Output shape: {outputs.shape}, Target shape: {labels.shape}")
                if CRITERION_NAME in MISC_FORMAT + N_D_N_FORMAT:
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

            def forward(self, outputs, labels):
                # Format labels into criterion-compatible
                formatted_outputs, formatted_labels = self.format_params(
                    outputs=outputs,
                    labels=labels
                )
    
                # Calculate normal criterion loss
                logging.debug(f"Labels type: {labels.shape} {labels.type()}")
                logging.debug(f"Formatted labels type: {formatted_labels.shape} {formatted_labels.type()}")
                loss = super().forward(formatted_outputs, formatted_labels)
                logging.debug(f"Criterion Loss: {loss.location}")

                # Add up all losses involved
                surrogate_loss = loss

                # Store result in cache
                self.__temp.append(surrogate_loss)
                
                return surrogate_loss

            def log(self):
                """ Computes mean loss across all current runs & caches the result """
                # log the mean loss for that specific worker only?
                avg_loss = th.mean(th.stack(self.__temp), dim=0)
                self._cache.append(avg_loss)
                self.__temp.clear()
                return avg_loss
            
            def reset(self):
                self.__temp = []
                self._cache = []
                return self

        return SurrogateCriterion

    # @staticmethod
    def parse_layers(self, model):
        model_layers = OrderedDict()
        for idx, (layer_name, a_func) in enumerate(model.layers.items()):
            curr_layer = getattr(model, layer_name) # model.nnl_2_linear == Linear(...)
            model_layers[layer_name] = curr_layer
            
            # To ignoring pre-applied activation function when creating layers and activations
            # This is to make slicing of L layers more intuitive during the creation of a global model.

            # Creation of layers and activations should have its own separate block/dict.
            # Layers and activations should be called from the torch.nn.modules
            # The global model structure should be defined in fedgkt format (See above for example on fedgkt model structure).

            func_name = a_func.__name__
            model_layers[f"{func_name}{idx}"] = a_func
            
        formatted_layers = OrderedDict({
            k:v for k,v in model_layers.items() 
            if '<lambda>' not in k
        })
        return formatted_layers


    def perform_FL_evaluation(self, datasets, workers=[], is_shared=True, **kwargs): 
        """ Obtains predictions given a validation/test dataset upon 
            a specified trained global model.
            
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
            logging.debug(f"packet: {packet}")

            worker, (data, labels) = packet
            logging.debug(f"Data: {data}, {type(data)}, {data.shape}")
            logging.debug(f"Labels: {labels}, {type(labels)}, {labels.shape}")
            logging.debug(f"Worker: {worker}, {type(worker)}")

            # for i in list(self.server_model.parameters()): # global_model
            #     logging.debug(f"Model parameters: {i}, {type(i)}, {i.shape}")

            # Skip predictions if filter was specified, and current worker was
            # not part of the selected workers
            if workers and (worker.id not in workers):
                return {}, None

            self.server_model = self.server_model.send(worker)
            self.local_models[worker.id] = self.local_models[worker.id].send(worker)

            self.server_model.eval()
            self.local_models[worker.id].eval()
            with th.no_grad():
                
                outputs, ef = self.local_models[worker.id](data)
                # outputs = outputs.detach()
                outputs = self.server_model(ef).detach()

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
                    raise ValueError(f"ML action {self.action} is not supported!")

                # Compute loss
                logging.debug(f"criterion_params: {self.arguments.criterion_params}")
                surrogate_criterion = self.build_custom_criterion()(
                    **self.arguments.criterion_params
                )
                
                loss = surrogate_criterion(
                    outputs=outputs, 
                    labels=labels,
                )
                logging.debug(f"outputs shape: {outputs.shape}")
                logging.debug(f"labels shape: {labels.shape}")
                # loss = nn.CrossEntropyLoss()(outputs, labels)
                


            self.local_models[worker.id] = self.local_models[worker.id].get()
            self.server_model = self.server_model.get() # if no get, will throw error?

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
            logging.debug(f"Batch: {batch}, {type(batch)}")

            batch_evaluations = {}
            batch_losses = []

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
                    raise RuntimeError("Feature data and label data are not in the same location!")
                packet = (data.location, batch)
                evaluated_worker_batch, loss = await evaluate_worker(packet)
                batch_evaluations.update(evaluated_worker_batch)
                batch_losses.append(loss)

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

    # Helper function to get computation shape
    def shape_after_alignment(self):
        """ Get the shape after alignment so the final output layer of the
            local model can be created automatically. 
        """
        shape = None
        for batch in self.train_loader:
            for worker, (data, labels) in batch.items():
                logging.debug(f"initialise_data_shape: {data.shape}")
                logging.debug(f"initialise_labels_shape: {labels.shape}")
                shape = (1, *(data.shape[1:])) # e.g. tabular (1, *(771, 28))
                break
            break
        return shape
    
    def initialise(self):
        model_structure = self.global_model
        shape = self.shape_after_alignment()
        logging.debug(f"initialise_shape: {shape}")
        L = self.L

        logging.debug(f"fedgkt model_structure: {model_structure}")
        logging.debug(f"criterion_params: {self.arguments.criterion_params}")

        local_models = self.generate_local_models(model_structure, L, shape)
        prev_models = self.generate_local_models(model_structure, L, shape)

        server_model = self.generate_server_model(model_structure, L)

        server_opt = self.arguments.optimizer( 
                **self.arguments.optimizer_params,
                params=server_model.parameters()
        )

        server_criterion = self.build_custom_criterion()(
                **self.arguments.criterion_params
            )
        
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
            stoppers,
            server_model,
            server_opt,
            server_criterion
        )


    def fit(self):
        logging.debug("fitting")
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

        # apply to server loss
        global_val_stopper = EarlyStopping(**self.arguments.early_stopping_params)

        rounds = 0
        pbar = tqdm(total=self.arguments.rounds, desc='Rounds', leave=True)
        while rounds < self.arguments.rounds:

            # logging.debug(f"Current global model:\n {self.global_model.state_dict()}")
            # logging.debug(f"Global Gradients:\n {list(self.global_model.parameters())[0].grad}")

            (
                local_models,
                prev_models, 
                optimizers, 
                schedulers, 
                criterions, 
                stoppers,
                server_model,
                server_opt,
                server_criterion
            ) = self.initialise()
            logging.debug("initialise done")

            # perform_parallel_training: return models, optimizers, schedulers, criterions, stoppers #, server_model, server_opt
            (retrieved_models, _, _, _, _) = self.perform_parallel_training(
                datasets=self.train_loader, 
                models=local_models,
                cache=prev_models,
                optimizers=optimizers, 
                schedulers=schedulers,
                criterions=criterions, 
                stoppers=stoppers,
                rounds=rounds,
                epochs=self.arguments.epochs,
                server_model=server_model,
                server_opt=server_opt,
                server_criterion=server_criterion
            )

            # Retrieve all models from their respective workers
            logging.debug(f"Current server model:\n {server_model.state_dict()}")
            self.server_model = server_model

            # logging.debug(f"Current global model:\n {self.global_model.state_dict()}")
            # aggregated_params = self.calculate_global_params(
            #     self.global_model, 
            #     retrieved_models, 
            #     self.train_loader
            # )

    #         # # Update weights with aggregated parameters 
    #         # self.global_model.load_state_dict(aggregated_params)
    #         # logging.debug(f"New global model:\n {self.global_model.state_dict()}")

            # Local losses for worker
            final_local_losses = {
                w.id: c._cache[-1].get()
                for w,c in criterions.items()
            }

            logging.debug(f'final_local_losses: {final_local_losses}')

            # Store local losses for analysis
            for w_id, loss in final_local_losses.items():
                local_loss_archive = self.loss_history['local'].get(w_id, {})
                local_loss_archive.update({rounds: loss.item()})
                self.loss_history['local'][w_id] = local_loss_archive

            # global_train_loss = th.mean(
            #     th.stack(list(final_local_losses.values())),
            #     dim=0
            # )

            global_train_loss = self.global_train_loss
            
            # Validate the global model
            _, evaluation_losses = self.evaluate(metas=['evaluate'])
            global_val_loss = evaluation_losses['evaluate']

            logging.debug(f"global_val_loss: {global_val_loss}")


            # # Store global losses for analysis
            global_loss_archive = self.loss_history['global']
            global_train_losses = global_loss_archive.get('train', {})
            global_train_losses.update({rounds: global_train_loss.item()})
            global_val_losses = global_loss_archive.get('evaluate', {})
            global_val_losses.update({rounds: global_val_loss.item()})
            self.loss_history['global'] = {
                'train': global_train_losses,
                'evaluate': global_val_losses
            }

            # If server model is deemed to have stagnated, stop training
            global_val_stopper(global_val_loss, self.server_model)
            if global_val_stopper.early_stop:
                logging.info("Global model has stagnated. Training round terminated!\n")
                break

            rounds += 1
            pbar.update(1)
        
        pbar.close()

    #     logging.debug(f"Objects in TTP: {self.crypto_provider}, {len(self.crypto_provider._objects)}")
    #     logging.debug(f"Objects in sy.local_worker: {sy.local_worker}, {len(sy.local_worker._objects)}")
        logging.debug(f"server logits size: {len(server_logits_dict)}")
        logging.debug(f"logits size: {len(logits_dict)}")
        logging.debug(f"labels size: {len(labels_dict)}")
        logging.debug(f"ef size: {len(extracted_feature_dict)}")

        return self.server_model, self.local_models #, server_model

    def generate_local_models(self, model_structure, L, shape) -> Dict[WebsocketClientWorker, sy.Plan]:
        """ Abstracts the generation of local models in a federated learning
            context. <-- insert your fedgkt local model definitions here -->

            IMPORTANT: 
            DO NOT distribute models (i.e. .send()) to local workers. Sending &
            retrieval have to be handled in the same functional context, 
            otherwise PySyft will have a hard time cleaning up residual tensors.

        Returns:
            Distributed context-specific local models (dict(str, Model))
        """
        # <-- Step 1: Slice out self.L layers to create a single local model first -->
        # <-- Step 2: Make copy of local models for N participants -->

        # create a model using Synergos model_structure format
        logging.debug(f"Generating local models..")
        # model = Model(structure=model_structure)
        worker_model_structure = self.parse_layers(model_structure)
        worker_model = WorkerModel(L, worker_model_structure, shape)
        logging.debug(f"Worker_model: {worker_model}")

        return {w: copy.deepcopy(worker_model) for w in self.workers}

    def generate_local_model(self, model_structure, L, shape):
        """ Abstracts the generation of local models in a federated learning
            context. <-- insert your fedgkt local model definitions here -->

            IMPORTANT: 
            DO NOT distribute models (i.e. .send()) to local workers. Sending &
            retrieval have to be handled in the same functional context, 
            otherwise PySyft will have a hard time cleaning up residual tensors.

        Returns:
            Distributed context-specific local models (dict(str, Model))
        """
        # <-- Step 1: Slice out self.L layers to create a single local model first -->
        # <-- Step 2: Make copy of local models for N participants -->

        # create a model using Synergos model_structure format
        logging.debug(f"Generating one local model..")
        # model = Model(structure=model_structure)
        worker_model_structure = self.parse_layers(model_structure)
        worker_model = WorkerModel(L, worker_model_structure, shape)
        logging.debug(f"Worker_model: {worker_model}")

        return copy.deepcopy(worker_model) 

    def generate_server_model(self, model_structure, L):
        """ Generation of the server model in a federated learning
            context.
        
        Args:
            model_structure: The original model structure
            L: sliced from L layer onwards
        """
        # This model is always global so no need to copy.
        logging.debug(f"Generating server model..")
        # model = Model(structure=model_structure)
        server_model_structure = self.parse_layers(model_structure)
        server_model = ServerModel(L, server_model_structure)
        logging.debug(f"server_model: {server_model}")
        return server_model

    
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
        server_model,
        server_opt,
        server_criterion
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
        Returns:
            trained local models
        """ 
        # Notes: pointer = pointer.get()

        # def insert_batch_info(my_dict, info, worker):
        #     batch_list = my_dict.get(worker, [])
        #     batch_list.append(info)
        #     my_dict[worker] = batch_list

        WORKERS_STOPPED = []
        async def train_worker(packet):
            """ Train a worker on its single batch, and does an in-place 
                updates for its local model, optimizer & criterion 
            
            Args:
                packet (dict):
                    A single packet of data containing the worker and its
                    data to be trained on 

            """ 
            """
            {
                'alice': [labels_0, labels_1, ....],
                'bob': [labels_0, labels_1, ....],
            }

            """
            worker, ((data, labels), batch_idx, server_logits_dict) = packet
            logging.debug(f"train_worker server_logits_dict: {server_logits_dict}")
            logging.debug(f"Data FedGKT: {data}, {type(data)}, {data.shape}")
            logging.debug(f"Labels: {labels}, {type(labels)}, {labels.shape}")

            # for i in list(self.global_model.parameters()):
            #     logging.debug(f"Model parameters: {i}, {type(i)}, {i.shape}")

            # Extract essentials for training
            # curr_global_model = cache[worker]
            curr_local_model = models[worker]
            curr_optimizer = optimizers[worker]
            curr_criterion = criterions[worker]

            # Check if worker has been stopped
            if worker.id not in WORKERS_STOPPED:

                # <-- USE THIS PART TO TRAIN UP LOCAL MODEL! -->
                # logging.debug(f"Before training FedGKT - Local Gradients for {worker}:\n {list(curr_local_model.parameters())[0].grad}")

                curr_local_model = curr_local_model.send(worker)
                # Zero gradients to prevent accumulation  
                curr_local_model.train()
                curr_optimizer.zero_grad()

                # Forward Propagation
                logging.debug(f"current model: {curr_local_model}")
                outputs, local_ef_ = curr_local_model(data)
                logging.debug(f"Output_data: {outputs.data}")
                logging.debug(f"local_ef_fp_shape: {local_ef_.shape}")
                logging.debug(f"local_ef_fp_location: {local_ef_.location}")
                logging.debug(f"local_ef_fp: {local_ef_}") # pointer tensor lost ???
                logging.debug(f"Data shape: {data.shape}")
                logging.debug(f"Output size: {outputs.shape}")
                logging.debug(f"Augmented labels size: {labels.shape}")

                local_loss = curr_criterion( # CE Loss
                    outputs=outputs, labels=labels)
                # local_loss = curr_criterion(outputs, labels)
                
                # Check if global variable server_logits_dict is empty
                if len(server_logits_dict) != 0:
                    # server_model_logits = server_logits_dict[worker.id][batch_idx]
                    logging.debug("server logits")
                    server_model_logits = server_logits_dict[worker.id][batch_idx].send(worker)
                    loss_kd = self.criterion_kl(outputs, server_model_logits)
                    loss = local_loss + self.alpha * loss_kd
                    server_model_logits = server_model_logits.get()             
                    logging.debug(f"server_model_logits: {server_model_logits}")

                else:
                    logging.debug("no server logits")
                    loss = local_loss

                # Backward propagation
                loss.backward()
                curr_optimizer.step()
                
                logging.debug(f"After training FedGKT - Local Gradients for {worker}:\n {list(curr_local_model.parameters())[0].grad}")

                # ### (EVAL) Storing local worker states #### 
                curr_local_model.eval()
                logging.debug(f"curr_local_model: {curr_local_model}")
                logging.debug(f"data: {data.shape}")
                logging.debug(f"curr_local_model_data: {curr_local_model(data)}")
                local_outputs, local_ef = curr_local_model(data)
                logging.debug(f"local_outputs.data: {local_outputs.data}")
                logging.debug(f"local_outputs: {local_outputs.shape}")

                logging.debug(f"local_ef.shape: {local_ef.shape}") # Object "66432032110" not found on worker when using tabular data?
                logging.debug(f"local_ef: {local_ef}") # Object "66432032110" not found on worker when using tabular data?


                # logging.debug(f"local_ef.data: {local_ef.data}")
                extracted_feature_dict[worker.id][batch_idx] = local_ef.get().data
                logits_dict[worker.id][batch_idx] = local_outputs.get().data
                labels_dict[worker.id][batch_idx] = labels.get().data
                # logging.debug(f"labels_dict_worker_id_batch_idx: {labels.data.clone().get()}")

                curr_local_model = curr_local_model.get()
                logging.debug(f"Finished evaluating local FedGKT")

            # Update all involved objects
            assert models[worker] is curr_local_model
            assert optimizers[worker] is curr_optimizer
            assert criterions[worker] is curr_criterion


        async def train_batch(batch, idx, server_logits_dict): # need batch_idx for the specific worker
            """ Asynchronously train all workers on their respective 
                allocated batches 

            Args:
                batch (dict): 
                    A single batch from a sliced dataset stratified by
                    workers and their respective packets. A packet is a
                    tuple pairing of the worker and its data slice
                    i.e. (worker, (data, labels))
            """
            # {
            #     worker: (data, labels),
            #     worker: ((data, labels), idx),
            #     worker: ((data, labels), idx),
            # }
            formatted_batch = {packet[0]: (packet[1], idx, server_logits_dict) for packet in batch.items()}
            for worker_future in asyncio.as_completed(
                map(train_worker, formatted_batch.items()) # 
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
                    ## empty arguments {} when using _retrieve_args on curr_scheduler.step
                    # step_args = self.arguments._retrieve_args(curr_scheduler.step)
                    step_args = list(inspect.signature(curr_scheduler.step).parameters)
                    logging.debug(f"step_args: {step_args}")
                    logging.debug(f"curr_scheduler: {curr_scheduler}")
                    if 'metrics' in step_args:
                        # e.g. The lrscheduler "ReduceLROnPlateau" requires 
                        # a metric to be passed in.
                        curr_scheduler.step(final_batch_loss)
                    else:
                        curr_scheduler.step()

            assert schedulers[worker] is curr_scheduler
            assert stoppers[worker] is curr_stopper 

        async def train_datasets(datasets):
            """ Train all batches in a composite federated dataset """
            # Note: All TRAINING must be synchronous w.r.t. each batch, so
            #       that weights can be updated sequentially!
            
            for idx, batch in enumerate(datasets):
                logging.debug("-"*90)
                # labels, logits, embeddings = await train_batch(batch, idx)
                await train_batch(batch, idx, server_logits_dict)         

            # require some tweak using earlystopping for fedgkt...
            logging.debug(f"Before stagnation evaluation: Workers stopped: {WORKERS_STOPPED}")
            stagnation_futures = [
                check_for_stagnation(worker) 
                for worker in self.workers
            ]
            await asyncio.gather(*stagnation_futures)
            logging.debug(f"After stagnation evaluation: Workers stopped: {WORKERS_STOPPED}")


        def train_server(whether_distill_on_the_server, server_epochs, 
                        model_server, server_opt, extracted_feature_dict,
                        logits_dict, labels_dict, server_criterion):
            """ Train your server model using embeddings obtained from local workers
                
            Args:
                whether_distill_on_the_server (bool): To perform knowledge distillation on Server
                server_epochs (int): No. of server epochs to perform
                model_server (nn.Module): server model
                server_opt (th.optim): server optimizer
                extracted_feature_dict (dict): computed feature map of inputs
                logits_dict (dict): computed logits from running all local workers model
                labels_dict (dict): original target labels from all workers dataloader
                server_criterion (th.nn): server criterion
            """
            server_logits_dict.clear()
            model_server.train()
            server_loss = 0
            client_dataset_size = 0

            # logging.debug(f"extracted_feature_dict: {extracted_feature_dict}")
            # extracted_feature_dict: {'test_participant_1': 
            # {0: (Wrapper)>[PointerTensor | ttp:88228605759 -> 
            # test_participant_1:22056785195]::data}, 
            # 'test_participant_2': {0: (Wrapper)>[PointerTensor | 
            # ttp:29744872742 -> test_participant_2:71164793366]::data}}

            logging.debug(f"model_Server: {model_server}")
            logging.debug("Start training server")
            for epoch in range(1, server_epochs + 1):
                for client_index in extracted_feature_dict.keys():
                    
                    # logging.debug(f"extracted_feature_dict_client: {extracted_feature_dict[client_index]}")
                    # extracted_feature_dict_client: {0: (Wrapper)>[PointerTensor | ttp:88228605759 
                    # -> test_participant_1:22056785195]::data}
                    extracted_feature_dict_ = extracted_feature_dict[client_index]
                    logits_dict_ = logits_dict[client_index]
                    labels_dict_ = labels_dict[client_index]

                    s_logits_dict = dict()
                    server_logits_dict[client_index] = s_logits_dict
                    for batch_index in extracted_feature_dict_.keys():
                        batch_feature_map_x = extracted_feature_dict_[batch_index]
                        batch_logits = logits_dict_[batch_index]
                        batch_labels = labels_dict_[batch_index]

                        output_batch = model_server(batch_feature_map_x)
 
                        # loss_true = nn.CrossEntropyLoss()(output_batch, batch_labels)
                        logging.debug(f"server_criterion_training: {server_criterion}")
                        loss_true = server_criterion(output_batch, batch_labels)

                        if whether_distill_on_the_server == 1:
                            loss_kd = self.criterion_kl(output_batch, batch_logits)
                            loss = loss_kd + self.alpha * loss_true
                        else:
                            loss = loss_true

                        server_loss += loss
                        server_opt.zero_grad()
                        loss.backward()
                        server_opt.step()

                        s_logits_dict[batch_index] = output_batch.detach()
                    client_dataset_size = client_dataset_size + len(extracted_feature_dict_)
                server_loss = server_loss / client_dataset_size
                # if epoch % 1 == 0:
                logging.debug('Server Epoch: {} Loss: {}'.format(epoch, server_loss))
            return model_server, server_loss
            
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

            # Is server considered a worker, since server also needs to train
            # declaring server model, optimizers and criterion? Same as local worker?
            logging.debug("Beginning to train server")
            logging.debug(f"workers list: {self.workers}")

            # Got error when running training w/o send..
            _, global_train_loss = train_server(
                whether_distill_on_the_server=self.whether_distill_on_the_server,
                server_epochs=self.server_epochs,
                model_server=server_model,
                server_opt=server_opt,
                extracted_feature_dict=extracted_feature_dict,
                logits_dict=logits_dict,
                labels_dict=labels_dict,
                server_criterion=server_criterion
            )
            logging.debug(f"global server logits dict: {server_logits_dict}")
            self.global_train_loss = global_train_loss
        finally:
            loop.close()


        return models, optimizers, schedulers, criterions, stoppers #, server_model, server_opt


    def evaluate(self, metas: List[str] = [],
             workers: List[str] = []) -> Tuple[Dict[str, Dict[str, th.Tensor]], Dict[str, th.Tensor]]:
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
        logging.debug(f"evaluate func:")
        DATA_MAP = {
            'train': self.train_loader,
            'evaluate': self.eval_loader,
            'predict': self.test_loader
        }
        
        # If no meta filters are specified, evaluate all datasets 
        metas = list(DATA_MAP.keys()) if not metas else metas

        # If no worker filter are specified, evaluate all workers
        workers = [w.id for w in self.workers] if not workers else workers

        # Evaluate global model using datasets conforming to specified metas
        inferences = {}
        losses = {}

        for meta, dataset in DATA_MAP.items():

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
        

    ##################
    # Core functions #
    ##################

    def analyse(self):
        """ Calculates contributions of all workers towards the final global 
            model. 
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

        def save_server_model():
            if 'global' in excluded: return None
            # Export server model (as the global component) to file
            server_model_out_path = os.path.join(
                out_dir, 
                "server_model.pt"
            )
            # Only states can be saved, since Model is not picklable
            if self.server_model != None:
                logging.debug(f"server_model_state_dict: {self.server_model.state_dict()}")
                th.save(self.server_model.state_dict(), server_model_out_path)
            else:
                server_model_out_path = ""
            return server_model_out_path

        out_paths = super().export(out_dir, excluded)

        # Package global metadata for storage
        out_paths['global']['path'] = save_server_model()

        return out_paths


    @staticmethod
    def combine_models(self, server_model, local_model):
        """
        """
        # combined model: local_model + server_model
        combined_model = None # Fill in yourself
        return combined_model


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
        logging.debug(f"restore function in: {archive}")
        for _type, logs in archive.items():

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

            logging.debug(f"restore parse global_model: {self.global_model}")

            if archived_origin == self.crypto_provider.id:
                logging.debug(f"server_archive_state: {archived_state}")
                # Generate server model (nn.Module) using self.global_model as a reference
                server_model = self.generate_server_model(self.global_model, self.L)
                # Now that the server model architecture has been initialised,
                # restore the archived server model's weights
                logging.debug(f"restored server_model: {server_model}")
                server_model.load_state_dict(archived_state)
                self.server_model = server_model
            
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
                    logging.debug(f"local_archive_state: {archived_state}")
                    # Need to determine shape by running through the batch
                    # since there is no way to determine the alignment shape
                    shape = self.shape_after_alignment()
                    logging.debug(f"local_archive_shape: {shape}")
                    # Generate local model (nn.Module) using self.global_model as a reference
                    archived_model = self.generate_local_model(self.global_model, self.L, shape)

                    # Now that the edge model architecture has been initialised,
                    # restore the archived edge model's weights
                    logging.debug(f"restored local_model: {archived_state}")
                    archived_model.load_state_dict(archived_state)
                
                self.local_models[archived_origin] = archived_model 



