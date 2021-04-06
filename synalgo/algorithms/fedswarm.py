#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
from collections import defaultdict

from tqdm import tqdm, tnrange, tqdm_notebook
from tqdm.notebook import trange

from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc, mean_squared_error, accuracy_score, roc_auc_score, average_precision_score, precision_recall_curve, log_loss, precision_score, recall_score, cohen_kappa_score

# Generic/Built-in
import asyncio
import copy
import json
import logging
import os
from collections import OrderedDict
from multiprocessing import Manager
from pathlib import Path
from typing import Tuple, List, Dict, Union
import time

# Libs
import numpy as np
import syft as sy
import torch as th
from syft.workers.websocket_client import WebsocketClientWorker

# Custom
from synalgo.arguments import Arguments
from synalgo.model import Model
from synalgo.algorithms.base import BaseAlgorithm

##################
# Configurations #
##################

model_hyperparams = {
    "binary": True,
    "regression": False,
    'swarm_size': 15,
    'batch_segment': None,
    'batches_per_member': 10,
    'with_replacement': False
}

##################################################
# Federated Algorithm Base Class - BaseAlgorithm #
##################################################

class FedSwarm(BaseAlgorithm):
    """
    Implements the FedSwarm algorithm.

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
        out_dir: str = '.'
    ):
        super().__init__(
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

        # Insert your own attributes here!
        self.out_dir = '.'
        self.checkpoints = defaultdict(dict)
        self.loss_history = defaultdict(dict)
        self.best_ensemble = []


    ############
    # Checkers #
    ############

    # Declare all helper functions that check on the current state of any
    # attribute in this section


    ###########
    # Setters #
    ###########

    # Declare all helper functions that modify custom attributes in this section


    ###########
    # Helpers #
    ###########


    def initialise(self):

        rounds = 0

        # was global_model at original code
        global_model = copy.deepcopy(self.global_model).send(self.crypto_provider)

        client_template = copy.deepcopy(self.global_model)

        local_models = {}
        optimizers = {}
        criterions = {}

        for worker in self.workers:
            w = worker.id
            client_models = []
            client_optimizers = []
            client_criterions = []
            for swarm_idx in range(model_hyperparams['swarm_size']):
                swarm_model = copy.deepcopy(client_template).send(w)

                swarm_optimizer = self.arguments.optimizer(
                    params=swarm_model.parameters(),
                    **self.arguments.optimizer_params
                )

                swarm_criterion = self.arguments.criterion()

                client_models.append(swarm_model)
                client_optimizers.append(swarm_optimizer)
                client_criterions.append(swarm_criterion)
            local_models.update({w: client_models})
            optimizers.update({w: client_optimizers})
            criterions.update({w: client_criterions})
            logging.debug("*******************************")
            logging.debug(f" Done initializing for worker {w}, {len(local_models)} local models")
            logging.debug("*******************************")

        return local_models, optimizers, criterions

    def train_swarm(self,
                  local_models,
                  optimizers,
                  criterions):
        trained_swarm = {}

        #pbar_workers = tqdm(total=len(self.workers), desc='Workers', leave=True)
        for idx, worker in enumerate(self.workers):
            worker_id = worker.id
            swarm_models = local_models[worker_id]
            swarm_optimizers = optimizers[worker_id]
            swarm_criterions = criterions[worker_id]

            trained_swarm_models = []

            self.checkpoints.update({worker.id:{}})
            self.loss_history.update({worker.id:{}})

            #pbar_workers = tqdm(total=len(swarm_models), desc=f'Traing Swarm Members of party {idx}', leave=True)
            for i in range(len(swarm_models)):

                trained_model = self.train_swarm_member(worker,
                                                        swarm_models[i],
                                                        swarm_optimizers[i],
                                                        swarm_criterions[i],
                                                        i,
                                                        len(swarm_models))

                #pbar_workers.update(1)

                trained_swarm_models.append(trained_model)
            trained_swarm.update({worker_id: trained_swarm_models})
            #pbar_workers.close()

            logging.debug("*******************************")
            logging.debug(f" Done training for worker {worker_id}")
            logging.debug("*******************************")


        logging.debug("*******************************")
        logging.debug(f" Done training for all parties")
        logging.debug("*******************************")
        return trained_swarm

    def train_swarm_member(self, worker, model, optimizer, criterion, swarm_idx, swarm_size):
        # pbar_batches = tqdm(total=len(self.train_loaders[worker]), desc=f'Batches {worker}', leave=False)
        curr_model = model
        curr_optimizer = optimizer
        curr_criterion = criterion

        #pbar_epochs = tqdm(total=self.arguments.epochs, desc=f'Epochs for swarm member {swarm_idx} of {worker}', leave=False)

        num_swarm_members = swarm_size

        num_batches = sum([1 for batch in self.train_loader if worker in batch.keys() ])
        batches_per_member = model_hyperparams['batches_per_member']
        # remainder = num_batches - (batches_per_member * num_swarm_members)
        batches = np.random.choice(range(num_batches), batches_per_member, replace=model_hyperparams['with_replacement'])

        logging.debug("*******************************")
        logging.debug(f" batchs ")
        logging.debug(batches)
        logging.debug("*******************************")
        print(batches)

        # self.checkpoints.update({worker.id:{}})
        # self.loss_history.update({worker.id:{}})

        for e in range(self.arguments.epochs):

            #pbar_batches = tqdm(total=batches_per_member, desc=f'Batches for swarm member {swarm_idx} of {worker}', leave=False)

            for batch_idx, batch in enumerate(self.train_loader):
                if batch_idx in batches:
                    data = batch[worker][0]
                    labels = batch[worker][1]
                    curr_model.train()
                    curr_optimizer.zero_grad()
                    predictions = curr_model(data.float())

                    # logging.debug('++++++++++++++++++++++++++++')
                    # logging.debug('PREDICTIONS')
                    # logging.debug(predictions.get())


                    if model_hyperparams['binary']:
                        loss = curr_criterion(predictions, labels.float())
                    else:
                        loss = curr_criterion(predictions, labels.long())
                    loss.backward()
                    curr_optimizer.step()

                    w = worker.id
                    self.checkpoints[w].update({
                        f'epoch_{e}_swarm_idx_{swarm_idx}':curr_model.get().state_dict()}
                    )

                    self.loss_history[w].update({
                        f'epoch_{e}_swarm_idx_{swarm_idx}': loss.get().item()
                    })

                    curr_model.send(worker)

        return curr_model

    # Declare all overridden or custom helper functions in this section


    ##################
    # Core functions #
    ##################

    # Override the 5 core functions `fit`, `evaluate`, `analyse`, `export` &
    # restore. Make sure that the class is self-consistent!

    def fit(self):
        """ Performs federated training using a pre-specified model as
            a template, across initialised worker nodes, coordinated by
            a ttp node.
        """
        round = 0
        local_models, optimizers, criterions = self.initialise()

        trained_swarm = self.train_swarm(local_models,
                                        optimizers,
                                        criterions)

        swarm = {}
        for w in trained_swarm.keys():
            worker_swarm_models = []
            for swarm_member in trained_swarm[w]:
                worker_swarm_models.append(swarm_member.get())
            swarm.update({w: worker_swarm_models})

        logging.debug('++++++++++++++++++++++++++++')
        logging.debug('testing evaluation function')
        logging.debug(swarm)

        self.local_models = swarm

        output_arr = []
        loss_arr = []
        auc_pr_arr = []
        for w in swarm:
            for model in swarm[w]:
                outputs, loss, auc_pr = self.perform_FL_evaluation_swarm(self.eval_loader, model)
                output_arr.append(outputs)
                loss_arr.append(loss)
                auc_pr_arr.append(auc_pr)


        logging.debug('++++++++++++++++++++++++++++')
        logging.debug('++++++++++++++++++++++++++++')
        logging.debug(output_arr)
        logging.debug(loss_arr)
        logging.debug(auc_pr_arr)
        naive_weights = [1/len(output_arr) for i in range(len(output_arr))]
        naive_auc_pr = self.calculate_ensemble_predictions(self.eval_loader,
                                                            output_arr,
                                                            naive_weights)

        logging.debug('++++++++++++++++++++++++++++')
        logging.debug('++++++++++++++++++++++++++++')
        logging.debug(naive_auc_pr)

        self.get_best_ensemble(auc_pr_arr, output_arr, swarm)

        logging.debug('++++++++++++++++++++++++++++')
        logging.debug(self.final_model_ids)
        logging.debug(self.final_weights)
        logging.debug(self.final_auc_pr)
        logging.debug(self.final_threshold)

        self.analyse()


    def threshold_arr(self,
                    auc_pr_arr,
                    prediction_arr,
                    model_ids,
                    THRESHOLD):
        new_arr1 = []
        new_arr2 = []
        new_arr3 = []
        threshold = np.mean(np.array(auc_pr_arr)) + (THRESHOLD * np.std(np.array(auc_pr_arr)))
        for i in range(len(auc_pr_arr)):
            if auc_pr_arr[i] > threshold:
                new_arr1.append(auc_pr_arr[i])
                new_arr2.append(prediction_arr[i])
                new_arr3.append(model_ids[i])
        return new_arr1, new_arr2, new_arr3

    def normalize_arr(self, arr):
        normalized = (arr - np.mean(arr)) / (np.std(arr) + 1e-10)
        return np.exp(normalized) / np.sum(np.exp(normalized), axis=0)

    def get_best_ensemble(self, auc_pr_arr, prediction_arr, swarm):

        best_auc_pr = 0
        best_performance = None
        best_models = None
        best_models_idxs = []
        best_APs = []
        best_THRESHOLD = 0

        model_ids = []

        for w in swarm:
            worker_name = f"{w}"
            for idx, model in enumerate(swarm[w]):
                model_ids.append(worker_name + '_' + str(idx))

        logging.debug('++++++++++++++++++++++++++++')
        logging.debug('Model ids')
        logging.debug(model_ids)

        for i in range(0, 5):
            THRESHOLD = 0 + (i * 0.1)
            threshold_model_AP, threshold_model_predictions, threshold_model_idxs = self.threshold_arr(auc_pr_arr,
                                                                                                    prediction_arr,
                                                                                                    model_ids,
                                                                                                    THRESHOLD)
            if len(threshold_model_predictions) < 1:
                pass

            logging.debug('++++++++++++++++++++++++++++')
            logging.debug('Sorted Model AUC_PRS')
            logging.debug(threshold_model_AP)
            logging.debug('Sorted model predictions')
            logging.debug(threshold_model_predictions)
            logging.debug('Sorted model ids')
            logging.debug(threshold_model_idxs)

            avg_distances = []
            for i in range(len(threshold_model_predictions)):
                avg_distance = 0
                for j in range(len(threshold_model_predictions)):
                    avg_distance += np.abs(np.sum(threshold_model_predictions[i] - threshold_model_predictions[j]) / len(threshold_model_predictions))
                avg_distance / len(threshold_model_predictions)
                avg_distances.append(avg_distance)

            threshold_model_AP, threshold_model_predictions, avg_distances, threshold_model_idxs = zip(*sorted(zip(threshold_model_AP,
                                                                                                                    threshold_model_predictions,
                                                                                                                    avg_distances,
                                                                                                                    threshold_model_idxs), reverse=True, key=lambda x: x[2]))

            logging.debug('++++++++++++++++++++++++++++')
            logging.debug('Sorted Model AUC_PRS by distance')
            logging.debug(threshold_model_AP)
            logging.debug('Sorted model predictions by distance')
            logging.debug(threshold_model_predictions)
            logging.debug('Sorted model ids by distance')
            logging.debug(threshold_model_idxs)
            logging.debug('Sorted distances by distance')
            logging.debug(avg_distances)

            for i in range(1, len(threshold_model_AP)):
                trimmed_model_AP = threshold_model_AP[0:i]
                trimmed_model_predictions = threshold_model_predictions[0:i]
                trimmed_model_idxs = threshold_model_idxs[0:i]
                norm_weights = self.normalize_arr(trimmed_model_AP)
                auc_pr = self.calculate_ensemble_predictions(self.eval_loader,
                                                            trimmed_model_predictions,
                                                            norm_weights)
                if auc_pr > best_auc_pr:
                    best_models_idxs = trimmed_model_idxs
                    best_weights = norm_weights
                    best_auc_pr = auc_pr
                    best_THRESHOLD = THRESHOLD

            self.final_model_ids = best_models_idxs
            self.final_weights = best_weights
            self.final_auc_pr = best_auc_pr
            self.final_threshold = best_THRESHOLD

    def calculate_auc_pr(self, labels, preds):
        logging.debug('++++++++++++++++++++++++++++')
        logging.debug('Calculating AUC-PR')
        logging.debug(labels)
        logging.debug(preds)
        pc_vals, rc_vals, _ = precision_recall_curve(labels, preds)
        logging.debug('Calculating AUC')
        logging.debug(pc_vals)
        logging.debug(rc_vals)
        logging.debug('++++++++++++++++++++++++++++')
        auc_pr = auc(rc_vals, pc_vals)
        logging.debug('Done')
        return auc_pr

    def calculate_ensemble_predictions(self, dataset, output_arr, weights):
        for idx, batch in enumerate(dataset):
            data = batch[0]
            labels = batch[1]
            worker = labels.location
            break

        logging.debug('++++++++++++++++++++++++++++')
        logging.debug('++++++++++++++++++++++++++++')
        logging.debug(output_arr)
        votes = np.zeros(np.array(output_arr[0].shape))
        for i in range(len(output_arr)):
                votes += weights[i] * output_arr[i]

        logging.debug('++++++++++++++++++++++++++++')
        logging.debug('++++++++++++++++++++++++++++')
        logging.debug(votes)

        votes = np.round(votes).astype(int)
        logging.debug(votes)
        logging.debug(weights)

        labels = labels.get()
        logging.debug('++++++++++Labels++++++++++')
        logging.debug(labels)
        detached_labels = labels.detach().numpy()
        logging.debug('++++++++Detached Labels+++++++++++')
        logging.debug(detached_labels)
        auc_pr = self.calculate_auc_pr(detached_labels, votes)
        logging.debug('Sending labels back')
        labels.send(worker)
        logging.debug('Sent')
        return auc_pr


    def perform_FL_evaluation_swarm(self, dataset, model, workers=[], is_shared=True, **kwargs):

        for idx, batch in enumerate(dataset):
            data = batch[0]
            labels = batch[1]
            worker = labels.location
            break

        logging.debug(f"Data: {data}, {type(data)}, {data.shape}")
        logging.debug(f"Labels: {labels}, {type(labels)}, {labels.shape}")

        model = model.send(worker)
        model.eval()
        with th.no_grad():
            outputs = model(data.float())

            outputs = outputs.get()
            labels = labels.get()

            loss = self.arguments.criterion()(outputs, labels.float())

            detached_labels = labels.detach().numpy()
            detached_outputs = np.round(outputs.detach().numpy()).astype(int)
            auc_pr = self.calculate_auc_pr(detached_labels, detached_outputs)

        model.get()
        loss = loss.item()
        labels.send(worker)
        return detached_outputs, loss, auc_pr

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

        raise NotImplementedError




    def analyse(self):
        """ Calculates contributions of all workers towards the final global
            model.
        """
        self.contri_matrix = {}

        contri_workers = [model[:-2] for model in self.final_model_ids]
        # test = [model for model in self.final_model_ids]
        # logging.debug('***************')
        # logging.debug('self.final_model_ids')
        # logging.debug(test)

        combined = list(sorted(zip(contri_workers, self.final_weights)))

        logging.debug('***************')
        logging.debug('combined')
        logging.debug(combined)

        res = defaultdict(list)
        for k, v in combined: res[k].append(v)
        logging.debug('res')
        logging.debug(res)
        for k in res.keys():
            self.contri_matrix[k] = sum(res[k])

        logging.debug('*************')
        logging.debug(f'obtained contri_matrix')
        logging.debug(self.contri_matrix)


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
        # """
        out_dir = out_dir if out_dir else self.out_dir

        out_paths = {}

        # Package global metadata for storage
        best_ensemble_model_paths, best_ensemble_weight_path = self.save_best_ensemble(out_dir)
        out_paths['global'] = {
            'origin': self.crypto_provider.id,
            'path': best_ensemble_model_paths,
            'weights_path': best_ensemble_weight_path
        }

        for idx, (worker_id, local_model) in enumerate(
            self.local_models.items(),
            start=1
        ):
            worker_model_path = self.save_worker_model(worker_id, out_dir)
            logging.debug("+++++++++++++++++++")
            logging.debug(worker_model_path)
            if isinstance(worker_model_path, list):
                pass
            else:
                worker_model_path = [worker_model_path]

            logging.debug("=================")
            logging.debug(worker_model_path)

            # Package local metadata for storage
            out_paths[f'local_{idx}'] = {
                'origin': worker_id,
                'path': worker_model_path,
                'loss_history': self.save_worker_losses(worker_id, out_dir),
            }
        return out_paths

    def save_best_ensemble(self, out_dir):

        best_ensemble_out_path = []
        # Only states can be saved, since Model is not picklable
        logging.debug(self.checkpoints)
        for i in self.final_model_ids:
            worker = i[:-2].strip()
            swarm_idx = i[-1:].strip()
            final_epoch = self.arguments.epochs-1
            final_epoch = f'epoch_{final_epoch}'
            curr_path = os.path.join(out_dir, f'global_model_{i}.pt')
            model_id = f'{final_epoch}_swarm_idx_{swarm_idx}'
            logging.debug(model_id + 'TEMP_MARKER')
            assert model_id in self.checkpoints[worker]
            th.save( self.checkpoints[worker][f'{final_epoch}_swarm_idx_{swarm_idx}'], curr_path)
            best_ensemble_out_path.append(curr_path)

        weights_path = os.path.join(out_dir, f'global_weights.json')
        with open(weights_path, 'w') as outfile:
            json.dump(self.contri_matrix, outfile)

        return best_ensemble_out_path, weights_path

    def save_worker_model(self, worker_id, out_dir):

        local_model_out_path = []

        for epoch_swarm, state in self.checkpoints[worker_id].items():

            state = self.checkpoints[worker_id][epoch_swarm]
            curr_path = os.path.join(out_dir, f'local_model_{worker_id}_{epoch_swarm}.pt')
            th.save(state, curr_path)
            local_model_out_path.append(curr_path)
        return local_model_out_path

    def save_worker_losses(self, worker_id, out_dir):
        local_loss_out_path = []
        for epoch_swarm in self.loss_history[worker_id]:
            curr_path = os.path.join(out_dir, f'local_loss_history_worker_{worker_id}_{epoch_swarm}.json')
            with open(curr_path, 'w') as llp:
                json.dump(self.loss_history.get(worker_id, {}), llp)
            local_loss_out_path.append(curr_path)
        return local_loss_out_path


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
            archive (dict): Dictionary containing versioned histories of
                exported filepaths corresponding to the state of models within a
                training cycle
            version (tuple(str)): A tuple where the first index indicates the
                round index and the second the epoch index
                (i.e. (round_<r_idx>, epoch_<e_idx>))
        """
        logging.debug(f'archive {archive}')
        for _, logs in archive.items():

            logging.debug(f"Logs: {logs}")
            archived_origin = logs['origin']

            all_models_path = logs['path']
            logging.debug(f'all_modesls_path is {all_models_path}')

            # Check if exact version of the federated grid was specified
            if version:
                round_idx = version[0]
                epoch_idx = version[1]
                logging.debug(f'version {version}')
                logging.debug(f'round_idx {round_idx}')
                logging.debug(f'epoch_idx {epoch_idx}')


                filtered_version = [i for i in all_models_path if i.startswith(f'local_model_epoch_{epoch_idx}')]
                logging.debug(f'filtered_version is {filtered_version}')
                archived_states = []
                for path in filtered_version:
                    logging.debug(f'path is {path}')
                    state = th.load(logs['checkpoints'][path])
                    archived_states.append(state)

            # Otherwise, load the final state of the grid
            else:
                archived_states = []
                logging.debug(f'logs {logs}')
                for p in logs['path']:
                    logging.debug(f'p {p}')
                if archived_origin == self.crypto_provider.id:

                    best_ensembles = [i for i in all_models_path if i.startswith(f'global_model_')]
                    for p in best_ensembles:
                        state = th.load(p)
                        archived_states.append(state)

                    for state in archived_states:
                        model = copy.deepcopy(self.global_model)
                        model.load_state_dict(state)
                        self.best_ensemble.append(model)

                else:

                    local_models = [i for i in all_models_path if (i.startswith(f'local_model_') and i.endswith('achived_origin'))]

                    for p in local_models:
                        state = th.load(p)
                        archived_states.append(state)

                    for state in archived_states:

                        model = copy.deepcopy(self.global_model)
                        model.load_state_dict(state)

                        self.local_models[archived_origin].append(model)
