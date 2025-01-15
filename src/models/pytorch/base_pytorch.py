#!/usr/bin/env python
# coding: utf-8

from src.models.base_model import BaseModel
from src.helpers.data import random_missing
import os
import pickle
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.helpers.utils_pytorch import CustomDataset
from sklearn.model_selection import ShuffleSplit
import pandas as pd
import inspect
from contextlib import redirect_stdout
import multiprocessing

import config
from src.helpers.utils_pytorch import df_to_torch_tensor
from src.helpers.utils import split_with_index

torch.set_num_threads(1)


class BaseModelPytorch(BaseModel, nn.Module):
    def __init__(self, path, name):
        BaseModel.__init__(self, path, name)
        nn.Module.__init__(self)
        self.device = torch.device(config.CUDA if torch.cuda.is_available() else 'cpu')
        self.study_name = None
        self.hp_storage = f"sqlite:///db.sqlite3"
        self.n_trials_per_process = None
        self.loader_workers = 0
        self.train_loader = None
        self.val_loader = None
        self.X_test = torch.Tensor().to(self.device)  # py tensor
        self.y_test = torch.Tensor().to(self.device)  # py tensor
        self.loss_func = nn.MSELoss()
        self.optimizer = None
        self.num_inputs = None
        self.input_size = None
        self.best_trial = None
        self.best_params = None
        self.model_version = None
        self.batch_size = 32
        self.EPOCHS = None

    def save_model(self):
        pickle.dump(self, open(os.path.join(self.output_dir, 'model.pkl'), 'wb'))
        return
        
    def create_task_dirs(self, task_key):
        self.split_path = os.path.join(self.split_path, task_key)
        if not os.path.exists(self.split_path):
            os.makedirs(self.split_path)

    def create_loader(self, split_index):
        features = self.X.loc[split_index][self.y.loc[split_index].notna().all(axis=1)]
        labels = self.y.loc[split_index].dropna()
        dataset = CustomDataset(features=features, labels=labels, device=self.device)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.loader_workers)

    def save_exp_def(self):
        with open(os.path.join(self.output_dir, 'experiment_def.txt'), 'w') as f:
            with redirect_stdout(f):
                print(inspect.getsource(self.define_hpspace))
                # print('EPOCHS = ', self.EPOCHS)
                print('MB-size = ', self.mb_size)
                print('INNER_SPLITS = ', config.INNER_SPLITS)
                print('OUTER_SPLITS = ', config.OUTER_SPLITS)
                print('OPTUNA_TRAILS = ', config.OPTUNA_TRAILS)
                print('Calc on = ', self.device)
                print('SEED = ', config.SEED)

    def evaluate(self, sparse_dict, i):
        if self.model_name in ['MMOE', 'FFNN_mtl', 'FFNN_input', 'MTLNET']:
            func = self.evaluate_base
        if self.model_name in ['FFNN_stl']:
            func = self.evaluate_stl

        # Option for Full data
        if sparse_dict['mode'] == 'full':
            y_pred = func()

        # Option for sparse-data
        elif sparse_dict['mode'] == 'sparse-all' and self.model_name in ['FFNN_input', 'FFNN_stl']:
            self.read_data()
            self.y_data.to_csv(os.path.join(self.split_path, 'y_data.csv'))
            self.y_data.loc[self.train_index] = random_missing(self.y_data.loc[self.train_index], sparse_dict['step'])
            self.y_data.to_csv(os.path.join(self.split_path, f'y_sparse-all.csv'))
            y_pred = func()

        # Option for sparse-data
        elif sparse_dict['mode'] == 'sparse-task' and self.model_name == 'FFNN_input':
            y_pred = pd.DataFrame()
            for y_label in self.y_label:
                self.read_data()
                self.y_data.loc[self.train_index, [y_label]] = random_missing(self.y_data.loc[self.train_index, [y_label]],
                                                                              sparse_dict['step'])
                self.create_task_dirs(y_label)
                # Train scaler and scale
                print('{:-^45}'.format(f'TASK {y_label}'))
                y_pred_task = self.evaluate_base()
                y_pred_task = y_pred_task[[y_label]]
                y_pred[y_label] = y_pred_task
                self.split_path = os.path.split(self.split_path)[0]
        else:
            print('Non possible combination of model and sparse data')
            y_pred = pd.DataFrame()
            pass

        return y_pred

    def evaluate_base(self):
        # Train scaler and scale
        self.scale()
        # Further data Preperation
        if self.model_name == 'FFNN_input':
            self.data_prepare_input()
        self.X.to_csv(os.path.join(self.output_dir, 'X_data.csv'), index=None)
        self.y.to_csv(os.path.join(self.output_dir, 'y_data.csv'), index=None)
        # HPO
        self.tune()
        # Retrain
        self.retrain()
        # Pred
        y_pred = self.predict()
        # Rescale preds
        y_pred.to_csv(os.path.join(self.split_path, 'preds.csv'), index=None)
        y_pred = self.rescale(y_pred)
        y_pred.to_csv(os.path.join(self.split_path, 'preds_rescaled.csv'), index=None)
        return y_pred

    def evaluate_stl(self):
        self.num_tasks = 1
        y_data = self.y_data[self.y_label].copy()
        y_pred = pd.DataFrame()
        for y_label in self.y_label:
            self.create_task_dirs(y_label)
            # Train scaler and scale
            self.y_data = y_data[[y_label]]
            print('{:-^45}'.format(f'TASK {y_label}'))
            y_pred_task = self.evaluate_base()
            y_pred[y_label] = y_pred_task
            self.split_path = os.path.split(self.split_path)[0]
        self.y_data = y_data
        self.num_tasks = self.y_data.shape[1]
        return y_pred

    def run_trail(self):
        study = optuna.load_study(study_name=self.study_name, storage=self.storage)
        study.optimize(self.objective, n_trials=self.n_trials_per_process)

    def tune(self):
        split_str = self.split_path[self.split_path.find('split'):].replace(os.sep, '_')
        self.study_name = f"{self.name}_{self.model_name}_{split_str}_{self.start_time}"
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            study_name=self.study_name,
            storage=self.hp_storage
        )

        self.n_trials_per_process = config.OPTUNA_TRAILS // config.NUM_CORES

        # Use multiprocessing for parallel execution
        processes = []
        for _ in range(config.NUM_CORES):
            p = multiprocessing.Process(target=self.run_trail)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        self.best_params = study.best_params

        best_score = study.best_value
        print(f"Best score: {best_score}\n")
        print(f"Optimized parameters: {self.best_params}\n")
        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        self.best_trial = study.best_trial

        print("  Value: {}".format(self.best_trial.value))

        print("  Params: ")
        for key, value in self.best_trial.params.items():
            print("    {}: {}".format(key, value))

        with open(os.path.join(self.split_path, f'best_params.txt'), 'w') as f:
            f.write(str(self.best_params))
            f.write(str(best_score))

    def objective(self, trial):
        # Hyperparameters
        params = self.define_hpspace(trial)
        inner_splits = split_with_index(self.train_index, n_splits=config.INNER_SPLITS, train_size=0.8)
        # Inner Loop CV

        cost_run = []

        for i, (split_train_index, split_val_index) in enumerate(inner_splits):
            # DF to Pytorchtensor
            self.train_loader = self.create_loader(split_train_index)
            self.val_loader = self.create_loader(split_val_index)
            # BaseModelPytorch
            self.build_model(params)
            self.to(self.device)
            # Training per Split
            self.train_()
            # Evaluation
            cost_split = 0
            self.eval()
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    predict = self(inputs)
                    for n in range(self.num_tasks):
                        cost_task = self.loss_func(predict[n].reshape(-1), labels[:, n])
                        cost_split = cost_split + cost_task
                cost_run.append(cost_split)
            trial.report(torch.mean(torch.stack(cost_run)), i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return torch.mean(torch.stack(cost_run))

    def train_(self):
        for step in range(self.EPOCHS):
            for X_batch, y_batch in self.train_loader:
                # forward
                self.optimizer.zero_grad()
                predict = self(X_batch)
                # compute loss
                loss = 0
                for n in range(self.num_tasks):
                    loss_task = self.loss_func(predict[n].reshape(-1), y_batch[:, n])
                    loss = loss + loss_task

                # backward
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

    def retrain(self):
        self.build_model(self.best_params)
        self.to(self.device)
        print(self.best_params)
        retrain_dataset = CustomDataset(features=self.X.loc[self.train_index][self.y.loc[self.train_index].notna().all(axis=1)],
                                        labels=self.y.loc[self.train_index].dropna(),
                                        device=self.device)
                                          
        self.train_loader = torch.utils.data.DataLoader(retrain_dataset, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=self.loader_workers)

        self.train_()

    def predict(self):
        self.X_test, self.y_test = df_to_torch_tensor(self.X.loc[self.test_index],
                                                      self.y.loc[self.test_index],
                                                      self.num_tasks,
                                                      self.device)
        self.eval()
        preds = pd.DataFrame()
        with torch.no_grad():
            predict = self(self.X_test)
            with open(os.path.join(self.split_path, f'split_test.txt'), 'w') as f:
                f.write(str(predict))
            if self.model_name == 'FFNN_input':
                preds = self.prepare_preds(predict)
                self.num_tasks = self.y_data.shape[1]
            else:
                for n in range(self.num_tasks):
                    if self.device == torch.device('cuda'):
                        preds[self.y_label[n]] = pd.Series(predict[n].cpu().numpy().flatten())
                    if self.device == torch.device('cpu'):
                        preds[self.y_label[n]] = pd.Series(predict[n].numpy().flatten())
        return preds
