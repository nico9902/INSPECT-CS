import argparse
from pathlib import Path
import os
import torch
import torch.nn 
import numpy as np
import random
import psutil

def build_loss(cfg):
    # get loss function
    if "loss" in cfg:
        loss_name = cfg.loss.loss_fn
        del cfg.loss.loss_fn
        loss_fn = getattr(torch.nn, loss_name)
        loss_function = loss_fn(**cfg.loss)
        return loss_function
    else:
        return None

def get_args():
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("-f", "--cfg_file", help="Path of Configuration File", type=str, required=True)
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=53667)
    parser.add_argument("--seed", type=int, default=42, help="Seed per la randomizzazione")
    parser.add_argument("--exp_name", type=str, default="experiment", help="Nome dell'esperimento")
    parser.add_argument("--task", type=str, required=True, help="Nome del task")
    parser.add_argument("--checkpoint", type=str, required=False, help="Checkpoint path per l'encoder dei reports nel caso di architettura gerarchica")

    return parser.parse_args()


def seed_all(seed):
    if not seed:
        seed = 0
    print("Using Seed : ", seed)

    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Empty and create direcotory
def create_dir(dir):
    if not os.path.exists(dir):
        Path(dir).mkdir(parents=True, exist_ok=True)


def delete_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass

def check_na(var):
    if var == "None":
        return None
    else:
        return var