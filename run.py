'''
Experiments on Caltech101-20 and NoisyMNIST.
'''

import os
import sys
# add directories in src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
# set cuda
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from applications.MvSCN import run_net
from core.Config import load_config
from core.data import get_data
import wandb

import tensorflow as tf

wandb.login(key="f4ee04bfcae66c9215ab791dd58659c92ff3d87f")
import functools

sweep_config = {
    'method': 'random',
    "metric": {'goal': 'minimize', 'name': 'loss'},
    'parameters': {
                'epochs': {'values': [20, 30]},
                'siam_k':{'values': [2,3,4,5,6,7,8,9, 10, 11, 12, 13, 14, 15, 16, 17, 18]},
                'lamb':{'values': [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001 , 0.0005, 0.00001]},
                }
    }
    
sweep_id = wandb.sweep(sweep_config, project="MvSCN-nus-sweep")


wandb.agent(sweep_id, run_net, count=15)

