import torch
import torch.nn as nn
import torch.optim as optim
from src.models.pytorch.base_pytorch import BaseModelPytorch


class MTLNET(BaseModelPytorch):
    def __init__(self, path, name):
        super().__init__(path, name)
        self.model_name = 'MTLNET'
        self.loss_func = nn.MSELoss()
        self.hidden_layers = nn.ParameterList(values=None)
        self.task_layers = nn.ParameterDict(parameters=None)
        self.model = None
        self.scheduler = None
        self.mb_size = 32

    @staticmethod
    def define_hpspace(trial):
        params = {'num_hidden': trial.suggest_int('num_hidden', 1, 2),
                  'num_neurons_hidden1': trial.suggest_int('num_neurons_hidden1', 5, 15),
                  'num_neurons_hidden2': trial.suggest_int('num_neurons_hidden2', 5, 15),
                  'num_hidden_task': trial.suggest_int('num_hidden_task', 1, 2),
                  'num_neurons_task-specific1': trial.suggest_int('num_neurons_task-specific1', 5, 15),
                  'num_neurons_task-specific2': trial.suggest_int('num_neurons_task-specific2', 5, 15),
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
        self.hidden_layers = nn.ParameterList(values=None)
        self.EPOCHS = params['epochs']

        # Input
        in_features = self.num_inputs

        # Hidden
        n = 1
        for i in range(params['num_hidden']):
            out_features = params[f'num_neurons_hidden{n}']
            self.hidden_layers.append(nn.Linear(in_features, out_features))
            self.hidden_layers.append(nn.Dropout(params['drop_out']))
            in_features = out_features
            n += 1
        output_shared_layers = out_features

        # Task specific
        for task in range(self.num_tasks):
            # Hidden - Task specific
            m = 1
            in_features = output_shared_layers
            self.task_layers[str(task)] = nn.ParameterList(values=None).to(self.device)
            for i in range(params['num_hidden_task']):
                out_features = params[f'num_neurons_task-specific{m}']
                self.task_layers[str(task)].append(nn.Linear(in_features, out_features))
                self.task_layers[str(task)].append(nn.Dropout(params['drop_out']))
                in_features = out_features
                m += 1
            # Output
            self.task_layers[str(task)].append(nn.Linear(in_features, 1))

        # Optimizer
        self.optimizer = getattr(optim, params['optimizer'])(self.parameters(),
                                                             lr=params['lr'],
                                                             weight_decay=params['weight_decay'])
                                                             
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=params['step_size'],
                                                         gamma=params['gamma'])

    def forward(self, x):
        y = []
        # Body
        for n in range(len(self.hidden_layers)):
            x = self.hidden_layers[n](x).to(self.device)
        # Heads
        for task in range(self.num_tasks):
            x_task = torch.clone(x).to(self.device)
            for n in range(len(self.task_layers[str(task)])):
                x_task = self.task_layers[str(task)][n](x_task).to(self.device)
            y.append(x_task)
        return y
