#!/usr/bin/env python

####################
# Required Modules #
####################


# Generic/Built-in
import abc

# Libs


# Custom


##################
# Configurations #
##################


#########################################
# FL Abstract Class - AbstractAlgorithm #
#########################################

class AbstractAlgorithm(abc.ABC):

    @abc.abstractmethod
    def fit(self):
        """ Performs federated training using a pre-specified model as
            a template, across initialised worker nodes, coordinated by
            a ttp node.
        """
        pass

    
    @abc.abstractmethod
    def evaluate(self):
        """ Obtains predictions given a validation/test dataset upon 
            a specified trained global model
        """
        pass
    

    @abc.abstractmethod
    def analyse(self):
        """ Calculates contributions of all workers towards the final global 
            model. 
        """
        pass
    

    def export(self, des_dir: str):
        """ Exports the global model state dictionary to file
        """
        pass


    def restore(self, archive: dict):
        """ Exports the global model state dictionary to file
        """
        pass
