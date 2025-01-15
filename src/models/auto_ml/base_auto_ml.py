#!/usr/bin/env python
# coding: utf-8


from src.models.base_model import BaseModel
from src.helpers.data import random_missing
import os
import re
from src.helpers.data import get_time_from_folder_name
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm, truncnorm
import collections
import pickle
from contextlib import redirect_stdout
import config


class BaseModelAutoML(BaseModel):
    def __init__(self, path, name):
        super().__init__(path, name)
        self.model = collections.defaultdict(dict)
        self.task_path = collections.defaultdict(dict)
        self.kde = collections.defaultdict(dict)
        self.stl_error = collections.defaultdict(dict)
        self.mapping = dict()
        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_test = pd.DataFrame()

    def create_task_dirs(self, modus, task_key):
        self.task_path[modus][task_key] = os.path.join(self.split_path, modus, task_key)
        if not os.path.exists(self.task_path[modus][task_key]):
            os.makedirs(self.task_path[modus][task_key])

    def save_model(self, modus, task_key):
        with open(os.path.join(self.task_path[modus][task_key], f'{self.model_name}_model.pickle'), 'wb') as f:
            pickle.dump(self.model[modus][task_key], f, pickle.HIGHEST_PROTOCOL)
        return

    def load_model(self, path, mode, task_key):
        print('Loaded Model from: ', path)
        print(f'Loaded Model mode: {mode}, Task: {task_key}')
        with open(os.path.join(path, task_key, f'{self.model_name}_model.pickle'), "rb") as input_file:
            self.model[mode][task_key] = pickle.load(input_file)

    def save_exp_def(self):
        with open(os.path.join(self.output_dir, 'experiment_def.txt'), 'w') as f:
            with redirect_stdout(f):
                print('INNER_SPLITS = ', config.INNER_SPLITS)
                print('OUTER_SPLITS = ', config.OUTER_SPLITS)
                print('NUM_CORES = ', config.NUM_CORES)
                print('MAX_TIME = ', config.MAX_TIME_MINUTES)
                print('MTL_LIST = ', config.MTL_LIST)
                print('SEED = ', config.SEED)

    def train_stl(self, y_label):
        # Train STL
        print('Dataset-Size')
        print(f'With NaN in Label {y_label}:', len(self.y_train[y_label]))
        print(f'WithOUT NaN in Label {y_label}:', len(self.y_train[y_label].dropna()))
        self.train('STL', y_label, self.X_train[self.y_train[y_label].notna()], self.y_train[y_label].dropna())
        self.save_model('STL', y_label)
        self.get_truncnorm('STL', y_label)
        with open(os.path.join(self.task_path['STL'][y_label], f'{self.model_name}_unc.pickle'), 'wb') as f:
            pickle.dump(self.stl_error[y_label], f, pickle.HIGHEST_PROTOCOL)

    def load_stl_models(self, y_label, sparse_dict, i):
        # get path to model and train index
        workdir = os.path.dirname(self.output_dir)
        list_runs = os.listdir(workdir)
        full_run = [x for x in list_runs if (self.name.split('_')[0]+'_' in x) &
                    ('full' in x) & ('_'+str(config.MAX_TIME_MINUTES)+'_min' in x)]
        sparse_run = [x for x in list_runs if (self.name.split('_')[0]+'_' in x) &
                      bool(re.search(f"sparse.*_{sparse_dict['step']}", x)) &
                      ('_'+str(config.MAX_TIME_MINUTES)+'_min' in x)]
        full_run = sorted(full_run, key=get_time_from_folder_name)
        sparse_run = sorted(sparse_run, key=get_time_from_folder_name)

        full_run_path = os.path.join(workdir, full_run[0], f'split_{i}', 'STL')
        sparse_run_path = os.path.join(workdir, sparse_run[0], f'split_{i}', 'STL')

        # check if current train_index is loaded train index
        loaded_train_index = np.loadtxt(os.path.join(workdir, full_run[0], f'split_{i}', 'train_index.txt'),
                                        delimiter=',')
        if np.array_equal(self.train_index, loaded_train_index):
            pass
        else:
            print('Models do not match train index')

        self.create_task_dirs('STL', '')

        # load model
        for task_key in [n for n in self.y_label if n != y_label]:
            self.load_model(full_run_path, 'STL', task_key)
            self.mapping[f'STL-Model_{task_key}'] = f'{full_run_path}_{task_key}_{self.model_name}_model'

        self.load_model(sparse_run_path, 'STL', y_label)
        self.mapping[f'STL-Model_{y_label}'] = f'{sparse_run_path}_{y_label}_{self.model_name}_model'
        # load trunc-norm
        for task_key in [n for n in self.y_label if n != y_label]:
            with open(os.path.join(full_run_path, task_key, f'{self.model_name}_unc.pickle'), "rb") as input_file:
                self.stl_error[task_key] = pickle.load(input_file)
            self.mapping[f'Truncnorm_{task_key}'] = f'{full_run_path}_{task_key}_{self.model_name}_unc.pickle'
        with open(os.path.join(sparse_run_path, y_label, f'{self.model_name}_unc.pickle'), "rb") as input_file:
            self.stl_error[y_label] = pickle.load(input_file)
        self.mapping[f'Truncnorm_{y_label}'] = f'{sparse_run_path}_{y_label}_{self.model_name}_unc.pickle'
        return

    def prep_train_mtl_pred(self, y_label):
        # Build MTL-dataset
        x_train_mtl = self.X_train.copy()
        x_train_stl = self.X_train.copy()
        if self.model_name == 'H2o':
            x_train_stl = self.prepare_data(x_train_stl, 'STL', y_label)
        # Build MTL-Data
        for stl_label in self.y_label:
            y_train_label = self.model['STL'][stl_label].predict(x_train_stl)
            if self.model_name == 'H2o':
                y_train_label = y_train_label.as_data_frame()
            x_train_mtl = pd.concat([x_train_mtl,
                                     pd.DataFrame(np.array(y_train_label),
                                                  index=x_train_mtl.index,
                                                  columns=[stl_label])],
                                    axis=1)
        x_train_mtl.replace(np.inf, x_train_mtl.mean(numeric_only=True), inplace=True)
        x_train_mtl.replace(-np.inf, x_train_mtl.mean(numeric_only=True), inplace=True)
        return x_train_mtl

    def get_mae(self):
        raise NotImplementedError("Please Implement this method in the Model-Class")

    def get_truncnorm(self, modus, task_key):
        mae = self.get_mae(modus, task_key)
        mean = 0
        a = 0

        def equation(sigma):
            return mean + sigma * (norm.pdf((a - mean) / sigma) / (1 - norm.cdf((a - mean) / sigma))) - mae

        sigma_solution = fsolve(equation, x0=1)
        self.stl_error[task_key] = truncnorm(a=-np.inf, b=np.inf, loc=mean, scale=sigma_solution)

    def prep_train_mtl_uncertainty(self, x_train_mtl):
        for stl_label in list(set(self.y_label) & (set(x_train_mtl.columns))):
            x_train_mtl[stl_label] = x_train_mtl[stl_label] + self.stl_error[stl_label].rvs(size=len(x_train_mtl),
                                                                                            random_state=config.SEED)
        return x_train_mtl

    def train_mtl(self, y_label, x_train_mtl, name):
        if name in ['MTL-true-other', 'MTL-predict-other', 'MTL-predict-other-unc']:
            x_train_mtl = x_train_mtl.drop([y_label], axis=1)
        self.train(name,
                   y_label,
                   x_train_mtl[self.y_train[y_label].notna()],
                   self.y_train[y_label].dropna())
        self.y_train[y_label].dropna().to_csv(os.path.join(self.task_path[name][y_label],
                                                           'y_train.csv'))
        x_train_mtl[self.y_train[y_label].notna()].to_csv(os.path.join(
            self.task_path[name][y_label], 'X_train_MTL.csv'))

    def predict(self, sparse_dict, i):
        self.X_test, self.y_test = self.X_data.loc[self.test_index], self.y_data.loc[self.test_index]
        preds = pd.DataFrame(index=self.y_test.index)
        # Pred STL
        for y_label in self.y_label:
            if sparse_dict['mode'] == 'sparse-task':
                self.load_stl_models(y_label, sparse_dict, i)
                pd.DataFrame(self.mapping, index=[0]).to_csv(os.path.join(self.task_path['STL'][y_label],
                                                                          'mapping.csv'), index=None)
            preds[f'{y_label}_STL'] = self.model['STL'][y_label].predict(self.X_test)

        # Pred MTL
        if any(x in ['MTL-true-other',
                     'MTL-predict-other', 'MTL-predict-all',
                     'MTL-predict-other-unc', 'MTL-predict-all-unc'] for x in config.MTL_LIST):
            for y_label in self.y_label:
                if sparse_dict['mode'] == 'sparse-task':
                    self.load_stl_models(y_label, sparse_dict, i)
                x_test = self.X_test.copy()
                for pred_STL in self.y_label:
                    x_label = self.model['STL'][pred_STL].predict(self.X_test)
                    x_test[pred_STL] = x_label
                x_test.replace([np.inf, -np.inf], np.nan, inplace=True)
                x_test.fillna(x_test.mean(), inplace=True)
                if 'MTL-predict-all' in config.MTL_LIST:
                    x_test.to_csv(os.path.join(self.task_path['MTL-predict-all'][y_label], 'x_test.csv'))
                    preds[f'{y_label}_MTL-predict-all'] = self.model['MTL-predict-all'][y_label].predict(x_test)
                if 'MTL-predict-all-unc' in config.MTL_LIST:
                    preds[f'{y_label}_MTL-predict-all-unc'] = self.model['MTL-predict-all-unc'][y_label].predict(x_test)
                x_test = x_test.drop([y_label], axis=1)
                x_test.to_csv(os.path.join(self.task_path['MTL-true-other'][y_label], 'x_test.csv'))
                if 'MTL-true-other' in config.MTL_LIST:
                    try:
                        preds[f'{y_label}_MTL-true-other'] = self.model['MTL-true-other'][y_label].predict(x_test)
                    except:
                        print('Error in MTL-true-other prediciton')
                        preds[f'{y_label}_MTL-true-other'] = 0
                if 'MTL-predict-other' in config.MTL_LIST:
                    preds[f'{y_label}_MTL-predict-other'] = self.model['MTL-predict-other'][y_label].predict(x_test)
                if 'MTL-predict-other-unc' in config.MTL_LIST:
                    preds[f'{y_label}_MTL-predict-other-unc'] = \
                        self.model['MTL-predict-other-unc'][y_label].predict(x_test)
            preds.replace([np.inf, -np.inf], np.nan, inplace=True)
            preds.fillna(preds.mean(), inplace=True)

        return preds

    def evaluate(self, sparse_dict, i):
        self.X_train, self.y_train = self.X_data.loc[self.train_index], self.y_data.loc[self.train_index]

        # Option for sparse-data 'sparse-all'
        if sparse_dict['mode'] == 'sparse-all':
            self.y_train = random_missing(self.y_train, sparse_dict['step'])
            self.y_train.to_csv(os.path.join(self.split_path, f'y_train_sparse-all.csv'))

        # Train STL
        if sparse_dict['mode'] in ['full', 'sparse-all']:
            for y_label in self.y_label:
                self.train_stl(y_label)

        # Train MTL
        for y_label in self.y_label:
            # Option for sparse-data 'sparse-task'
            if sparse_dict['mode'] == 'sparse-task':
                self.create_task_dirs('STL', y_label)
                self.load_stl_models(y_label, sparse_dict, i)
                self.y_train = self.y_data.loc[self.train_index]
                self.y_train[[y_label]] = random_missing(self.y_train[[y_label]], sparse_dict['step'])
                self.y_train[[y_label]].to_csv(os.path.join(self.split_path, f'y_train_{y_label}_MTL.csv'))

            if 'MTL-true-other' in config.MTL_LIST:
                x_train_mtl = pd.concat([self.X_train, self.y_train], axis=1)
                self.train_mtl(y_label, x_train_mtl, 'MTL-true-other')
            if 'MTL-predict-other' in config.MTL_LIST:
                x_train_mtl = self.prep_train_mtl_pred(y_label)
                self.train_mtl(y_label, x_train_mtl, 'MTL-predict-other')
            if 'MTL-predict-other-unc' in config.MTL_LIST:
                x_train_mtl = self.prep_train_mtl_pred(y_label)
                x_train_mtl = self.prep_train_mtl_uncertainty(x_train_mtl)
                self.train_mtl(y_label, x_train_mtl, 'MTL-predict-other-unc')

            if 'MTL-predict-all' in config.MTL_LIST:
                x_train_mtl = self.prep_train_mtl_pred(y_label)
                self.train_mtl(y_label, x_train_mtl, 'MTL-predict-all')
            if 'MTL-predict-all-unc' in config.MTL_LIST:
                x_train_mtl = self.prep_train_mtl_pred(y_label)
                x_train_mtl = self.prep_train_mtl_uncertainty(x_train_mtl)
                self.train_mtl(y_label, x_train_mtl, 'MTL-predict-all-unc')

        y_pred = self.predict(sparse_dict, i)
        y_pred.to_csv(os.path.join(self.split_path, 'preds.csv'), index=None)
        if self.model_name == 'MLjar':
            self.clean_dir()
        if self.model_name == 'H2o':
            self.clean_mem()
            self.clean_tmp()
        return y_pred
