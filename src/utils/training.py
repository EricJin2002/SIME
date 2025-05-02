import os
import torch
import numpy as np
import torch.distributed as dist
import matplotlib.pyplot as plt
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def plot_history(train_history, num_epochs, ckpt_dir, seed):
    # save training curves
    plt.figure()
    plt.plot(np.linspace(0, num_epochs, len(train_history)), train_history, label = 'train')
    plt.tight_layout()
    plt.legend()
    plt.title("loss")
    plt.savefig(os.path.join(ckpt_dir, f'train_seed_{seed}.png'))

def plot_history_v2(train_history, num_epochs, ckpt_dir, fig_name):
    # save training curves
    plt.figure()
    plt.plot(np.linspace(0, num_epochs, len(train_history)), train_history, label = 'train')
    plt.tight_layout()
    plt.legend()
    plt.title("loss")
    plt.savefig(os.path.join(ckpt_dir, fig_name))

def sync_loss(loss, device):
    t = [loss]
    t = torch.tensor(t, dtype = torch.float64, device = device)
    dist.barrier()
    dist.all_reduce(t, op = torch.distributed.ReduceOp.AVG)
    return t[0]