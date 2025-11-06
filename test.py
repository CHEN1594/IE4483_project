# ==============================
# 0) Setup and Global Config
# ==============================
import torch, sys, os
print("Torch version:", torch.__version__)
print("Torch file:", torch.__file__)


import os, random, numpy as np, torch

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Relative paths (based on current working directory)
PROJECT_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJECT_DIR, "datasets")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
WEIGHT_PTH = os.path.join(MODEL_DIR, "efficientnet_b3_imagenet_pretrained.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", DEVICE)

