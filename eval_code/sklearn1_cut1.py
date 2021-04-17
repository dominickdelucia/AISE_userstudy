import pandas as pd
import numpy as np
from functools import *
import sys
from hypertuner import HyperTuner


# This class should be the main workhorse for model training
# Each Learner should have a learnertype, params, list of features,and a bogey name (?)
# Each learner should output a predictor object and a list of training keys (timestamps)
class Learner():

    def __init__(self, learnertype, params, Splitter, Scaler, Sampler, HyperGrid=None, CV=10, scorer=None):
        self.trained = False
        self.feat_list = None
        self.bogey = None

        self.learnertype = learnertype
        self.params = params
        self.hypergrid = HyperGrid
        self.cv = CV
        if self.hypergrid is None:
            self.obj = learnertype(**params)
        else:
            self.obj = learnertype()

        self.splitter = Splitter
        self.scaler = Scaler
        self.sampler = Sampler
        self.scorer = scorer
        pass

    def ingest_augmentor(self, augmentor, raw_df):
        self.augmentor = augmentor
        self.augmentor.augmented_df = self.augmentor.