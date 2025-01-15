# Model
MODEL = 'XGBoost'

# Data
DATA_FOLDER = 'data/run'
NAME_DATA = 'data.csv'

# Options: 'full', 'sparse-all', 'sparse-task'
SPARSE_DICT = {'mode': ['full', 'sparse-all', 'sparse-task'],
               'steps': [0.75, 0.5, 0.25]}

# CV
INNER_SPLITS = 10
DEFAULT_TRAIN_SIZE = 0.8
OUTER_SPLITS = 10

# Pytorch
# Computing
OPTUNA_TRAILS = 200

# AutoML
# Computing
NUM_CORES = 8
MAX_TIME_MINUTES = 15
# Options: 'MTL-true-other', 'MTL-predict-other', 'MTL-predict-all',  'MTL-predict-other-unc', 'MTL-predict-all-unc'
MTL_LIST = ['MTL-true-other', 'MTL-predict-other', 'MTL-predict-all',  'MTL-predict-other-unc', 'MTL-predict-all-unc']
SEED = 1

# Framework Specific
H2O_PORT = 54321  # Important to set different Ports when running H2o in parallel


# Model library
model_dict = {'MMOE': {'dir': 'pytorch', 'script': 'mmoe', 'class': 'MMOE'},
              'FFNN_mtl': {'dir': 'pytorch', 'script': 'ffnn_mtl', 'class': 'FFNN'},
              'FFNN_input': {'dir': 'pytorch', 'script': 'ffnn_mtl', 'class': 'FFNN'},
              'FFNN_stl': {'dir': 'pytorch', 'script': 'ffnn_mtl', 'class': 'FFNN'},
              'MTLNET': {'dir': 'pytorch', 'script': 'mtlnet', 'class': 'MTLNET'},
              'AutoSklearn': {'dir': 'auto_ml', 'script': 'autosklearn', 'class': 'AutoSklearn'},
              'H2o': {'dir': 'auto_ml', 'script': 'h2o', 'class': 'H2o'},
              'MLjar': {'dir': 'auto_ml', 'script': 'mljar', 'class': 'MLjar'},
              'XGBoost': {'dir': 'auto_ml', 'script': 'xgboost', 'class': 'XGBoost', 'version': '-'}}
