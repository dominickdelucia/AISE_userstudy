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
            self.xy_data[key]['_y'] = self.split_data[key].loc[:, self.bogey]
        pass

    def x_scale(self):
        tr_key = pd.Series(self.xy_data.keys())[pd.Series(self.xy_data.keys()).str.contains('sample')].values[0]
        self.scaler.fit_scaler(self.xy_data[tr_key]['_X'])
        for k in self.xy_data.keys():
            self.xy_data[k]['_X'] = self.scaler.scale(self.xy_data[k]['_X'])
        pass

    def sample_train_data(self):
        self.split_data['sampled'] = self.sampler.sample(self.split_data['train'])
        pass

    def train(self):
        self.actual_training_data = self.xy_data['train']['_X']
        self.pred_obj = self.obj.fit(self.actual_training_data, self.xy_data['train']['_y'])
        self.trained = True
        self.used_data = self.split_data['train']['timestamp'].values
        pass

    # TODO: need to set up evaluation here
    def get_test_data(self):
        pass

    def hypertune(self):
        self.hypertuner = HyperTuner(self, self.hypergrid, self.cv, self.scorer)
        self.hypertuner.tune()
        return (self.hypertuner.get_optimal_params())

    def get_predictor(self):
        self.predictor = Predictor(self.pred_obj, self.scaler, self.feat_list)
        return (self.predictor)


class Predictor():

    def __init__(self, pred_obj, scaler, feat_list):
        self.obj = pred_obj
        self.scaler = scaler
        self.feat_list = feat_list
        pass

    def get_feats(self, df):
        all_cols = df.columns
        bool_col_idx = [elem in all_cols for elem in self.feat_list]
        allthere = all(bool_col_idx)
        if not allthere:
            return ('error: not all feature columns available')
        # check that all feats are in the available columns
        # return an error if all feats aren't available
        else:
            # then return the filtered df if they all are available
            return df.loc[:, self.feat_list]

    def predict(self, df):
        feat_df = self.get_feats(df)
        scaled_df = self.scaler.scale(feat_df)
        return self.obj.