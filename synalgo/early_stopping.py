#!/usr/bin/env python

####################
# Required Modules #
####################

# Generics
import logging
import os

# Libs
import numpy as np
import torch as th
#from ignite.engine import Engine, Events
#from ignite.handlers import EarlyStopping

# Custom


##################
# Configurations #
##################


############################################
# Model optimisation Class - EarlyStopping #
############################################

class EarlyStopping:
    """ 
    Early stops the training if validation loss doesn't improve after a given 
    patience.

    Args:
        patience (int): last time validation loss improved
                        (Default: 10)
        delta  (float): Minimum change required to qualify as an improvement
                        (Default: 0)
        verbose (bool): Toggles message for each validation loss improvement
                        (Default: False)

    Attributes:
        patience   (int): last time validation loss improved
        delta    (float): Minimum change required to qualify as an improvement
        verbose   (bool): Toggles message for each validation loss improvement
        counter    (int): Tracks no. of epochs passed since last improvement
        best_score (int): Tracks the lowest score obtained
        early_stop    (bool): Marker for when training should stop
        val_loss_min (float): Tracks the minimum lowest validation loss obtained 
    """
    def __init__(self, patience=10, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        

    def __call__(self, val_loss, model):
        """ Evaluates current validation loss calculated 
        
        Args:
            val_loss  (float): Computed loss to be evaluated
            model (nn.Module): Model to be evaluated
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    ###########
    # Helpers #
    ###########

    def save_checkpoint(self, val_loss, model):
        """ Saves model when validation loss decrease
        """
        if self.verbose:
            logging.debug(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...',
            )

        # th.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

##############
# Deprecated #
##############
"""
# Early Stopping parameters
self.patience = patience, 
self.score_function = score_function
self.trainer = trainer
self.min_delta = min_delta
self.cumulative_delta = cumulative_delta
"""
"""
# Store the best model
def default_score_fn(engine):
    score = engine.state.metrics['Accuracy']
    return score


best_model_handler = ModelCheckpoint(dirname=log_path,
                                     filename_prefix="best",
                                     n_saved=3,
                                     global_step_transform=global_step_from_engine(trainer),
                                     score_name="test_acc",
                                     score_function=default_score_fn)
evaluator.add_event_handler(Events.COMPLETED, best_model_handler, {'model': model, })

# Add early stopping
es_patience = 10
es_handler = EarlyStopping(patience=es_patience, score_function=default_score_fn, trainer=trainer)
evaluator.add_event_handler(Events.COMPLETED, es_handler)
setup_logger(es_handler._logger)


# Clear cuda cache between training/testing
def empty_cuda_cache(engine):
    torch.cuda.empty_cache()
    import gc
    gc.collect()


trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)
train_evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)

trainer.run(train_loader, max_epochs=num_epochs)
"""