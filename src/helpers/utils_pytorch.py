import numpy as np
import torch
from torch.utils.data import Dataset
import config
import math


torch.cuda.manual_seed_all(config.SEED)


def numpy_to_pytorch(input_):
    output = torch.from_numpy(input_).float()
    return output


def random_mini_batches(X_batch, y_batch, n_task, mini_batch_size=10):
    # Creating the mini-batches
    # np.random.seed(seed)
    m = X_batch.shape[0]                  
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X_batch[permutation, :]
    shuffled_y = []
    for n in range(n_task):
        shuffled_y.append(y_batch[n][permutation])
    num_complete_minibatches = math.floor(m/mini_batch_size)

    for k in range(0, int(num_complete_minibatches)):
        mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size, :]
        mini_batch_y = []
        for n in range(n_task):
            mini_batch_y.append(shuffled_y[n][k*mini_batch_size:(k+1)*mini_batch_size])
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)

    lower = int(num_complete_minibatches * mini_batch_size)
    upper = int(m - (mini_batch_size * math.floor(m/mini_batch_size)))

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[lower: lower + upper, :]
        mini_batch_y = []
        for n in range(n_task):
            mini_batch_y.append(shuffled_y[n][lower: lower + upper])
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)

    return mini_batches


def numpy_to_pytorch(input_):
    output = torch.from_numpy(input_).float()
    return output


def df_to_torch_tensor(X, y, num_tasks, device):
    # Convert to numpy_array
    X = X.to_numpy()
    y_list = []

    for n in range(num_tasks):
        y_list.append(y.iloc[:, n].to_numpy())

    # Convert numpy_array to torch tensor
    X = numpy_to_pytorch(X)
    X = X.to(device)
    for n in range(num_tasks):
        y_list[n] = numpy_to_pytorch(y_list[n])
        y_list[n] = y_list[n].to(device)

    return X, y_list
    
    
# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, features, labels, device):
        self.features = torch.tensor(features.values, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels.values, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y
