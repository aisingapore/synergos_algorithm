#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic
import json
import logging
import os
import random
import subprocess
from collections import defaultdict, OrderedDict
from glob import glob
from pathlib import Path
from string import Template

# Libs
import numpy as np
import psutil
import torch as th

# Custom


##################
# Configurations #
##################

API_VERSION = "0.2.0"

####################
# Helper Functions #
####################

def seed_everything(seed=42):
    """ Convenience function to set a constant random seed for model consistency

    Args:
        seed (int): Seed for RNG
    Returns:
        True    if operation is successful
        False   otherwise
    """
    try:
        random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        return True

    except:
        return False
