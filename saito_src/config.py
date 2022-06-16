import torch
import transformers
import os

ROOT_DIR = '../input/foursquare-location-matching'
OUT_DIR = '../output/0616_fold0/'

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

DEBUG = False

seed = 42
n_neighbors = 3 if DEBUG else 10
n_splits = 4
one_fold = True if DEBUG else True


fold_to_train=0


# Training config
train_batch_size = 256
valid_batch_size = 256
epochs = 25
lr = 5e-5
n_accumulate = 1
max_grad_norm = 1000
weight_decay = 1e-6

# TO DO: move this away from config
device = torch.device('cuda')

# Model config
model_name = "xlm-roberta-large"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
max_length = 32

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
