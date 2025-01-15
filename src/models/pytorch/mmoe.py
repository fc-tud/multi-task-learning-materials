import torch
import torch.nn as nn
import torch.optim as optim
from src.models.pytorch.base_pytorch import BaseModelPytorch


class MMOE(BaseModelPytorch):
    def __init__(self, path, name):
        super().__init__(path, name)
        self.model_name = 'MMOE'
        self.loss_func = nn.MSELoss()
        self.num_experts = None
        self.hidden_neu_expert = None
        self.num_neurons_expert = None
        self.num_neurons_expert2 = None
        self.hidden_tower_neu = None
        self.num_neurons_tower = None
        self.scheduler = None
        self.mb_size = 32

    @staticmethod
    def define_hpspace(trial):
        params = {'num_experts': trial.suggest_int('num_experts', 1, 10),
                  'hidden_neu_expert': trial.suggest_int('hidden_neu_expert', 1, 25),
                  'hidden_tower_neu': trial.suggest_int('hidden_tower_neu', 1, 20),
                  'num_neurons_expert': trial.suggest_int('num_neurons_expert', 1, 50),
                  'num_neurons_expert2': trial.suggest_int('num_neurons_expert2', 1, 20),
                  'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop"]),
                  'lr': trial.suggest_float("lr", 1e-5, 1e-1, log=True),
                  'step_size': trial.suggest_int("step_size", 10, 1000),
                  'gamma': trial.suggest_float("gamma", 0.1, 1),
                  'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True),
                  'drop_out': trial.suggest_float("drop_out", 0, 0.5),
                  'epochs': trial.suggest_int("epochs", 50, 1000),
                  }
        return params

    def build_model(self, params):
        self.num_neurons_tower = 1
        self.EPOCHS = params['epochs']
        # Experts
        for i in range(params['num_experts']):
            setattr(self, 'expert'+str(i), nn.Sequential(
                nn.Linear(self.num_inputs, params['num_neurons_expert']),
                nn.LeakyReLU(),
                nn.Linear(params['num_neurons_expert'], params['hidden_neu_expert']),
                nn.Dropout(params['drop_out']),
                nn.ReLU(),
                nn.Linear(params['hidden_neu_expert'], params['num_neurons_expert2']),
                nn.Dropout(params['drop_out']),
                nn.LeakyReLU()
            ))
        # Gates
        for i in range(self.num_tasks): 
            setattr(self, 'gate'+str(i), nn.Sequential(
                nn.Linear(self.num_inputs, params['num_experts']),
                nn.Softmax(dim=1)
            ))
        # Towers
        for i in range(self.num_tasks):
            setattr(self, 'tower'+str(i), nn.Sequential(
                nn.Linear(params['num_neurons_expert2'], params['hidden_tower_neu']),
                nn.Dropout(params['drop_out']),
                nn.LeakyReLU(),
                nn.Linear(params['hidden_tower_neu'], self.num_neurons_tower),
            )) 

        # For forward call
        self.num_experts = params['num_experts']
        self.num_neurons_expert2 = params['num_neurons_expert2']

        # Optimizer
        self.optimizer = getattr(optim, params['optimizer'])(self.parameters(),
                                                             lr=params['lr'],
                                                             weight_decay=params['weight_decay'])

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=params['step_size'],
                                                         gamma=params['gamma'])

    def forward(self, xv):
        bs = xv.shape[0]
        # experts
        out_experts = torch.zeros(self.num_experts, bs, self.num_neurons_expert2).to(self.device)
        for i in range(self.num_experts):
            out_experts[i] = getattr(self, 'expert'+str(i))(xv)
        # gates and weighted opinions
        input_towers = torch.zeros(self.num_tasks, bs, self.num_neurons_expert2).to(self.device)
        for i in range(self.num_tasks):
            gate = getattr(self, 'gate'+str(i))(xv)
            for j in range(self.num_experts):
                input_towers[i] += gate[:, j].unsqueeze(dim=1)*out_experts[j]
        # towers
        out_towers = torch.zeros(self.num_tasks, bs, self.num_neurons_tower).to(self.device)
        for i in range(self.num_tasks):
            out_towers[i] = getattr(self, 'tower'+str(i))(input_towers[i])
        output = torch.sigmoid(out_towers).to(self.device)
        return output
