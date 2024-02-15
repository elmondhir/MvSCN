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

wandb.login(key="f4ee04bfcae66c9215ab791dd58659c92ff3d87f")


# load config for NoisyMNIST 
config = load_config('./config/noisymnist.yaml')

# load config for Caltech101-20
# config = load_config('./config/Caltech101-20.yaml')

# use pretrained SiameseNet. 
config['siam_pre_train'] = True

# LOAD DATA
data_list = get_data(config)


# siam_lrs= [0.0001, 0.001, 0.01]



# RUN EXPERIMENT
wandb.init(
project="MvSCN",
name=f"MvSCN-best",
)

print(f"MvSCN-best")
# print(get_run_name("MvSCN", run))
x_final_list, scores = run_net(data_list, config)

wandb.finish()

