#!/usr/bin/env python
# coding: utf-8

import os
import itertools
import config
from config import MODEL
from src.helpers.utils import create_work_dir
from src.helpers.data import check_dataset
from src.models.ncv_bench import ncv_bench
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sparse_list = list(itertools.product(config.SPARSE_DICT['mode'], config.SPARSE_DICT['steps'], repeat=1))
sparse_list = [x if x[0] != 'full' else ('full', 1) for x in sparse_list]
sparse_list = list(dict.fromkeys(sparse_list))

if __name__ == '__main__':
    for dataset in os.listdir(config.DATA_FOLDER):
        for sparse_mode in sparse_list:
            path = os.path.join(config.DATA_FOLDER, dataset)
            files = check_dataset(path)
            print(79 * 'v')
            print(f'Dataset name: {dataset}')
            sparse_dict = {'mode': sparse_mode[0], 'step': sparse_mode[1]}
            print(f"Sparse mode: {sparse_dict['mode']}, Data available: {sparse_dict['step']}")
            if config.model_dict[MODEL]['dir'] == 'auto_ml':
                resources = f'{config.MAX_TIME_MINUTES}_min'
            if config.model_dict[MODEL]['dir'] == 'pytorch':
                resources = f'{config.OPTUNA_TRAILS}_trails'
            name = os.path.basename(path) + '_' + sparse_dict['mode'] + '_' + str(sparse_dict['step']) + '_' \
                                                                            + resources
            # check for done tasks
            work_dir = create_work_dir(MODEL)
            done_tasks = ['_'.join(x.split('_')[:5]) for x in os.listdir(work_dir)]
            if name in done_tasks:
                continue
            if os.path.basename(path) == 'xiong' and sparse_mode[1] < 0.5:
                continue
            ncv_bench(path, name, MODEL, sparse_dict)

    print(79 * '^')
