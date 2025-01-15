#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from contextlib import redirect_stdout
from datetime import datetime

from src.models.auto_ml.base_auto_ml import BaseModelAutoML
from src.helpers.utils import truncate_log_file
import config

import sklearn.metrics as metrics

import h2o
from h2o.automl import H2OAutoML


class H2o(BaseModelAutoML):
    def __init__(self, path, name):
        super().__init__(path, name)
        self.model_name = 'H2o'
        self.val_preds = pd.DataFrame()

    def prepare_data(self, data, modus, task_key):
        # h2o needs to read data in its own format, so
        # 1. save data into csv
        # 2. import data csvs in h2o
        # 3. remove csv
        data.to_csv(os.path.join(self.task_path[modus][task_key], 'temp_csv.csv'), index=False)
        h2o_data = h2o.import_file(os.path.join(self.task_path[modus][task_key], 'temp_csv.csv'))
        for col in h2o_data.columns:
            h2o_data[col] = h2o_data[col].asnumeric()
        return h2o_data

    def train(self, modus, task_key, X_data, y_data):
        self.create_task_dirs(modus, task_key)

        # --- FRAMEWORK SPECIFIC START --- #1 -------------------------------------
        h2o.init(nthreads=config.NUM_CORES,
                 min_mem_size='40g',
                 port=config.H2O_PORT,
                 log_level='ERRR',
                 max_log_file_size='2MB')

        data = X_data.merge(y_data, left_index=True, right_index=True, suffixes=['_x', ''], how='outer')
        h2o_data = self.prepare_data(data, modus, task_key)

        self.model[modus][task_key] = H2OAutoML(max_runtime_secs=config.MAX_TIME_MINUTES * 60,
                                                nfolds=config.INNER_SPLITS,
                                                sort_metric="MSE",
                                                # distribution=self.distribution[metric_key],
                                                keep_cross_validation_predictions=True,
                                                seed=config.SEED)
        # --- FRAMEWORK SPECIFIC END --- #1 -------------------------------------
        print(datetime.now())
        print(f'Training in {modus} of {task_key} over {config.MAX_TIME_MINUTES} min started')
        x_list = h2o_data.columns
        x_list.remove(y_data.name)
        self.model[modus][task_key].train(x=x_list,
                                          y=y_data.name,
                                          training_frame=h2o_data)
        print(datetime.now())
        
        lb = self.model[modus][task_key].leaderboard
        print(lb.head(rows=10))  # print all, not only default (10)

        best_model = self.model[modus][task_key].get_best_model()
        print(best_model)
        with open(os.path.join(self.task_path[modus][task_key], 'best_model.txt'), 'w') as f:
            with redirect_stdout(f):
                print(best_model)

        if (modus == 'STL') and (best_model.cross_validation_holdout_predictions()):
            self.val_preds = pd.DataFrame()
            self.val_preds['target'] = y_data
            self.val_preds['prediction'] = best_model.cross_validation_holdout_predictions().as_data_frame()
            self.val_preds['prediction'].fillna(self.val_preds['prediction'].mean(), inplace=True)
        else:
            self.val_preds = pd.DataFrame(columns=['target', 'prediction'], index=[0])
            cv_results = best_model.cross_validation_metrics_summary().as_data_frame()
            self.val_preds['target'] = 0
            self.val_preds['prediction'] = cv_results['mean'].loc[cv_results[''] == 'mae'].item()

    def predict(self, sparse_dict, i):
        self.X_test, self.y_test = self.X_data.loc[self.test_index], self.y_data.loc[self.test_index]
        preds = pd.DataFrame(index=self.y_test.index)

        # Pred STL
        for y_label in self.y_label:
            if sparse_dict['mode'] == 'sparse-task':
                self.load_stl_models(y_label, sparse_dict, i)
                self.create_task_dirs('STL', y_label)
                pd.DataFrame(self.mapping, index=[0]).to_csv(os.path.join(self.task_path['STL'][y_label],
                                                                          'mapping.csv'), index=None)
            h2o_test_data = self.prepare_data(self.X_test, 'STL', y_label)
            preds[f'{y_label}_STL'] = np.array(self.model['STL'][y_label].predict(h2o_test_data).as_data_frame())

        # Pred MTL
        if any(x in ['MTL-true-other',
                     'MTL-predict-other', 'MTL-predict-all',
                     'MTL-predict-other-unc', 'MTL-predict-all-unc'] for x in config.MTL_LIST):
            for y_label in self.y_label:
                if sparse_dict['mode'] == 'sparse-task':
                    self.load_stl_models(y_label, sparse_dict, i)
                # Define STL input in MTL-Model
                x_test = self.X_test.copy()
                stl_labels = [n for n in self.y_label if n != y_label]
                for pred_STL in stl_labels:
                    h2o_test_data = self.prepare_data(self.X_test, 'MTL-predict-other', y_label)
                    x_label = self.model['STL'][pred_STL].predict(h2o_test_data).as_data_frame()
                    x_test[pred_STL] = np.array(x_label)
                x_test.replace(np.inf, x_test.mean(numeric_only=True), inplace=True)
                x_test.replace(-np.inf, x_test.mean(numeric_only=True), inplace=True)
                if 'MTL-true-other' in config.MTL_LIST:
                    h2o_test_data = self.prepare_data(x_test, 'MTL-true-other', y_label)
                    preds[f'{y_label}_MTL-true-other'] = np.array(self.model['MTL-true-other'][y_label].
                                                                  predict(h2o_test_data).as_data_frame())
                if 'MTL-predict-other' in config.MTL_LIST:
                    preds[f'{y_label}_MTL-predict-other'] = np.array(self.model['MTL-predict-other'][y_label].
                                                                     predict(h2o_test_data).as_data_frame())
                if 'MTL-predict-other-unc' in config.MTL_LIST:
                    preds[f'{y_label}_MTL-predict-other-unc'] = np.array(self.model['MTL-predict-other-unc'][y_label].
                                                                         predict(h2o_test_data).as_data_frame())

                if 'MTL-predict-all' in config.MTL_LIST:
                    h2o_test_data = self.prepare_data(self.X_test, 'MTL-predict-other', y_label)
                    x_label = self.model['STL'][y_label].predict(h2o_test_data).as_data_frame()
                    x_test[f'{y_label}_x'] = np.array(x_label)

                    # format test data to h2o format
                    # test_data = X_test.merge(self.y_test[y_label], left_index=True, right_index=True, how='outer')
                    h2o_test_data = self.prepare_data(x_test, 'MTL-predict-all', y_label)
                    preds[f'{y_label}_MTL-predict-all'] = np.array(self.model['MTL-predict-all'][y_label].
                                                                   predict(h2o_test_data).as_data_frame())
                if 'MTL-predict-all-unc' in config.MTL_LIST:
                    preds[f'{y_label}_MTL-predict-all-unc'] = np.array(self.model['MTL-predict-all-unc'][y_label].
                                                                       predict(h2o_test_data).as_data_frame())

        return preds

    def get_mae(self, modus, task_key):
        self.val_preds.to_csv(os.path.join(self.task_path[modus][task_key], 'val_preds.csv'))
        mae = metrics.mean_absolute_error(self.val_preds['target'], self.val_preds['prediction'])
        return mae

    # specific save and load function for H2o
    def save_model(self, modus, task_key):
        best_model = self.model[modus][task_key].get_best_model()
        best_model.save_mojo(path=os.path.join(self.task_path[modus][task_key]),
                             force=True, filename=f'{self.model_name}_model')

    def load_model(self, path, mode, task_key):
        h2o.init(nthreads=config.NUM_CORES, min_mem_size='40g', port=config.H2O_PORT)
        self.model[mode][task_key] = h2o.import_mojo(os.path.join(path, task_key, f'{self.model_name}_model'))

    @staticmethod
    def clean_mem():
        h2o.remove_all()

    @staticmethod
    def clean_tmp(self, base_path='/tmp'):
        temp_dir_list = [d for d in os.listdir(base_path) if
                         'tmp' in d and os.path.isdir(os.path.join(base_path, d))]
        for dir_ in temp_dir_list:
            dir_path = os.path.join(base_path, dir_)
            file_list = [f for f in os.listdir(dir_path) if f.startswith('h2o') and f.endswith('.out')]
            for file_name in file_list:
                file_path = os.path.join(dir_path, file_name)
                truncate_log_file(file_path)
