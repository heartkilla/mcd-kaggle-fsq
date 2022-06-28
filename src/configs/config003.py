import os

import torch
import transformers

ROOT_DIR = '../input/foursquare-location-matching'
OUT_DIR = f'../output/{os.path.basename(__file__).replace(".py", "")}/'

INPUT = 'train_filled_revgeo_v1.csv'

if not os.path.exists(OUT_DIR):
   os.makedirs(OUT_DIR)

DEBUG = False

seed = 42
n_neighbors = 3 if DEBUG else 10
n_splits = 4
one_fold = True if DEBUG else True

# Training config
train_batch_size = 128
valid_batch_size = 128
epochs = 40
lr = 5e-5
n_accumulate = 4
max_grad_norm = 1000
weight_decay = 1e-6
warmup_epochs = 4
gradient_checkpointing = True

# TO DO: move this away from config
device = torch.device('cuda')

# Model config
model_name = "xlm-roberta-large"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
max_length = 64
simple_mean = False

# Metric loss and its params
loss_module = 'arcface'
s = 30.0
m_start = 0.2 
m_end = 0.6
ls_eps = 0.0
easy_margin = False

# Model parameters
n_classes = 739972  # df["point_of_interest"].nunique() = 739972
pooling = 'clf'
use_fc = False
dropout = 0.0
fc_dim = 320
