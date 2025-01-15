import torch
import torch.nn as nn
import torch.optim as optim
from src.models.pytorch.base_pytorch import BaseModelPytorch
import pandas as pd
import numpy as np
import os


class FFNN(BaseModelPytorch):
    def __init__(self, path, name):
        super().__init__(path, name)
        self.loss_func = nn.MSELoss()
        self.layers = nn.ParameterList(values=None)
        self.model = None
        self.scheduler = None
        self.mb_size = 32

    @staticmethod
    def define_hpspace(trial):
        params = {'num_hidden': trial.suggest_int('num_hidden', 1, 3),
                  'num_neurons_hidden1': trial.suggest_int('num_neurons_hidden1', 5, 35),
                  'num_neurons_hidden2': trial.suggest_int('num_neurons_hidden2', 5, 35),
                  'num_neurons_hidden3': trial.suggest_int('num_neurons_hidden3', 5, 35),
                  'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                  'lr': trial.suggest_float("lr", 1e-5, 1e-1, log=True),
                  'step_size': trial.suggest_int("step_size", 10, 1000),
                  'gamma': trial.suggest_float("gamma", 0.1, 1),
                  'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True),
                  'drop_out': trial.suggest_float("drop_out", 0, 0.5),
                  'epochs': trial.suggest_int("epochs", 50, 1000),
                  }
        return params

    def build_model(self, params):
        self.layers = nn.ParameterList(values=None)
        self.EPOCHS = params['epochs']

        # Input
        in_features = self.num_inputs

        # Hidden
        n = 1
        for i in range(params['num_hidden']):
            out_features = params[f'num_neurons_hidden{n}']
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.Dropout(params['drop_out']))
            in_features = out_features
            n += 1

        # Output
        self.layers.append(nn.Linear(in_features, self.num_tasks))

        # Optimizer
        self.optimizer = getattr(optim, params['optimizer'])(self.parameters(),
                                                             lr=params['lr'],
                                                             weight_decay=params['weight_decay'])

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=params['step_size'],
                                                         gamma=params['gamma'])

    def forward(self, x):
        y = []
        for n in range(len(self.layers)):
            x = self.layers[n](x).to(self.device)
        for task in range(self.num_tasks):
            y_task = x.index_select(1, torch.tensor([task]).to(self.device))
            y.append(y_task)
        return y

    def data_prepare_input(self):
        self.num_inputs = self.X_data.shape[1]
        self.num_inputs += self.num_tasks
        self.num_tasks = 1
        df = pd.DataFrame()
        for y in self.y_label:
            self.X.loc[:, f'X_{y}'] = 0
        for y in self.y_label:
            df_temp = pd.concat([self.X, self.y[[y]].rename(columns={y: 'Y'})], axis=1)
            df_temp[f'X_{y}'] = 1
            df = pd.concat([df, df_temp], axis=0)
        df.to_csv(os.path.join(self.split_path, 'df.csv'))
        df = pd.read_csv(os.path.join(self.split_path, 'df.csv'), index_col=0)
        df.to_csv(os.path.join(self.split_path, 'df_1.csv'))
        self.X = df.drop('Y', axis=1)
        self.y = df[['Y']]

    def prepare_preds(self, predict):
        if self.device == torch.device('cuda'):
            predict = pd.Series(predict[0].cpu().numpy().flatten())
        if self.device == torch.device('cpu'):
            predict = pd.Series(predict[0].numpy().flatten())
        df = self.X.loc[self.test_index].copy()
        df['Y'] = np.array(predict)
        result = pd.DataFrame()
        for y in self.y_label:
            result[y] = df['Y'].loc[np.array(df[f'X_{y}'] == 1)]
        return result
