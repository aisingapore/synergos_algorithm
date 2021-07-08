#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
from typing import Tuple, List, Dict, Union

# Libs
import syft as sy
from syft.frameworks.torch.fl.dataloader import FederatedDataLoader

# Custom


##################
# Configurations #
##################


############################################################
# Main Custom Dataloader class - CustomFederatedDataloader #
############################################################

class CustomFederatedDataloader(FederatedDataLoader):
    """ 
    A custom dataloader to override PySyft's default FederatedDataLoader.
    Instead of terminating iteration in accordance to the dataset with the
    lowest batch size, it iterates until all batches of the largest dataset are
    processed.

    Arguments:
        federated_dataset (sy.FederatedDataset): dataset from which to load the data
        batch_size (int): how many samples per batch to load (default: 1)
        shuffle (bool): Toggles if data is reshuffled at every epoch (default: False)
        num_iterators (int): number of workers from which to retrieve data in 
            parallel. num_iterators <= len(federated_dataset.workers) - 1, the 
            effect is to retrieve num_iterators epochs of data but at each step 
            data from num_iterators distinct workers is returned.
    """

    __initialized = False

    def __init__(
        self,
        federated_dataset: sy.FederatedDataset,
        batch_size: int = 8,
        shuffle: bool = False,
        num_iterators: int = 1
    ):
        super().__init__(
            federated_dataset=federated_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,            # so that no inferences get lost
            iter_per_worker=True,       # for LVL 1A parallelization
            num_iterators=num_iterators
        )


    def __next__(self) -> dict:
        """ Modified iterator behaviour to extract & process maximum no. of 
            batches across all participants

        Returns:
            Participants' batches (dict)
        """
        batches = {}
        terminations = 0
        for iterator in self.iterators:
            
            try:
                data, target = next(iterator)
                batches[data.location] = (data, target)
            except StopIteration:
                terminations += 1

        # Every cached iterator has been iterated through completely
        if terminations == len(self.iterators):
            raise StopIteration

        return batches
