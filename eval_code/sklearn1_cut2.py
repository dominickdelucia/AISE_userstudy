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
        self.augmentor.augmented_df = self.augmentor.make_augmented_df(raw_df)
        self.augmented_df = self.augmentor.get_augmented_df()
        self.augmentor.feat_list = list(self.augmentor.feature_identifier(self.augmented_df).columns)
        self.feat_list = self.augmentor.feat_list
        self.bogey = self.augmentor.bogey
        pass

    def make_split_data(self):
        self.split_data = self.splitter.split(self.augmented_df)
        pass

    def xy_separate(self):
        self.xy_data = dict()
        for key in pd.Series(self.split_data.keys()):
            self.xy_data[key] = dict()
            self.xy_data[key]['_X'] = self.split_data[key].loc[:, self.feat_list]
            self.xy_data[key]['_y'] = self.split_data[key].loc[:, self.