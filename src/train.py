import os
import sys
import time
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.distributed as dist

from tqdm import tqdm
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

from utils.training import set_seed, plot_history_v2

def train(cfg):

    # prepare distributed training
    torch.multiprocessing.set_sharing_strategy('file_system')
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    RANK = int(os.environ['RANK'])
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    os.environ['NCCL_P2P_DISABLE'] = '1'
    dist.init_process_group(backend = 'nccl', init_method = 'env://', world_size = WORLD_SIZE, rank = RANK)

    # set up device
    set_seed(cfg.seed)
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert torch.cuda.is_available()

    if RANK==0:
        log_root = os.path.join(cfg.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        original_stdout = sys.stdout
        sys.stdout = open(os.path.join(log_root, 'log.txt'), 'w')

    # dataset
    if RANK == 0: print("Loading dataset ...")
    if cfg.dataset.name == "realworld":
        if cfg.policy.name == "DP" or cfg.policy.name == "BC":
            from dataset.realworld import RealWorldDataset, collate_fn, pre_process_data
        else:
            raise NotImplementedError
        dataset = RealWorldDataset(**cfg.dataset.params)
    elif cfg.dataset.name == "robomimic":
        from dataset.robomimic_v2 import RoboMimicDataset, collate_fn, pre_process_data
        dataset = RoboMimicDataset(**cfg.dataset.params)
    elif cfg.dataset.name == "robomimic_old":
        from dataset.robomimic import RoboMimicDataset, collate_fn, pre_process_data
        dataset = RoboMimicDataset(**cfg.dataset.params)
    else:
        raise NotImplementedError
    print("dataset size:", len(dataset))
    
    # sampler
    if not hasattr(cfg, "num_iters_per_epoch") or cfg.num_iters_per_epoch is None:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            num_replicas = WORLD_SIZE, 
            rank = RANK, 
            shuffle = True
        )
    else:
        num_samples = cfg.num_iters_per_epoch * cfg.batch_size
        # random sampler
        sampler = torch.utils.data.RandomSampler(
            dataset, 
            replacement = False,  # make sure all data are used before some are reused
            num_samples = num_samples
        )
        from utils.distributed import DistributedSamplerWrapper
        sampler = DistributedSamplerWrapper(
            sampler, 
            num_replicas = WORLD_SIZE, 
            rank = RANK
        )

    # dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = cfg.batch_size // WORLD_SIZE,
        num_workers = cfg.num_workers,
        collate_fn = collate_fn,
        sampler = sampler
    )

    # load policy
    if RANK == 0: print("Loading policy ...")
    if cfg.policy.name == "DP":
        from policy.dp import DP
        policy = DP(**cfg.policy.params).to(device)
    elif cfg.policy.name == "BC":
        from policy.bc import get_bc
        policy = get_bc(cfg.policy.params.algo)(**cfg.policy.params).to(device)
    elif cfg.policy.name == "ResBC":
        from policy.residual_bc import ResBC
        policy = ResBC(**cfg.policy.params).to(device)
    else:
        raise NotImplementedError
    if RANK == 0:
        n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))
    policy = nn.parallel.DistributedDataParallel(
        policy, 
        device_ids = [LOCAL_RANK], 
        output_device = LOCAL_RANK, 
        find_unused_parameters = True
    )

    # load checkpoint
    if cfg.resume_ckpt is not None:
        policy.module.load_state_dict(torch.load(cfg.resume_ckpt, map_location = device), strict = False)
        if RANK == 0:
            print("Checkpoint {} loaded.".format(cfg.resume_ckpt))

    # ckpt path
    if RANK == 0:
        ckpt_dir = os.path.join(log_root, "ckpt")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
    
    # optimizer and lr scheduler
    if RANK == 0: print("Loading optimizer and scheduler ...")
    if cfg.optimizer.name == "AdamW":
        optimizer = torch.optim.AdamW(policy.parameters(), **cfg.optimizer.params)
    else:
        raise NotImplementedError

    if cfg.lr_scheduler.name == "cosine_with_warmup":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer = optimizer,
            num_warmup_steps = 2000,
            num_training_steps = len(dataloader) * cfg.num_epochs
        )
    else:
        raise NotImplementedError
    lr_scheduler.last_epoch = len(dataloader) * (cfg.resume_epoch + 1) - 1

    # save config
    if RANK == 0:
        with open(os.path.join(log_root, 'config.json'), 'w') as f:
            json.dump(cfg, f, indent = 4)

    # training
    train_history = {}

    policy.train()
    for epoch in range(cfg.resume_epoch + 1, cfg.num_epochs):
        if RANK == 0: print("Epoch {}".format(epoch)) 
        sampler.set_epoch(epoch)
        optimizer.zero_grad()
        num_steps = len(dataloader)
        pbar = tqdm(dataloader) if RANK == 0 else dataloader
        avg_loss = {"loss": 0.0}
        
        for data in pbar:
            batch = pre_process_data(data, device, cfg)
            # forward
            loss = policy(**batch)
            if isinstance(loss, dict):
                for key in loss:
                    if key not in avg_loss:
                        avg_loss[key] = 0.0
                    avg_loss[key] += loss[key].item()
                loss = loss["loss"]
            else:
                avg_loss["loss"] += loss.item()
            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        # avg_loss = avg_loss / num_steps
        for key in avg_loss:
            avg_loss[key] /= num_steps
        
        # Synchronize avg_loss across all processes
        for key in avg_loss:
            tensor = torch.tensor(avg_loss[key], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            avg_loss[key] = tensor.item()
        
        for key in avg_loss:
            if key not in train_history:
                train_history[key] = []
            train_history[key].append(avg_loss[key])

        if RANK == 0:
            print("Train loss: {:.6f}".format(avg_loss["loss"]), end = " ")
            for key in train_history:
                print("{}: {:.6f}".format(key, train_history[key][-1]), end = " ")
            print("")
            if (epoch + 1) % cfg.save_epochs == 0 \
                or (epoch + 1) in list(range(cfg.num_epochs - 100, cfg.num_epochs, 10)): # save last 100 epochs every 10 epochs
                torch.save(
                    policy.module.state_dict(),
                    os.path.join(ckpt_dir, "policy_epoch_{}_seed_{}.ckpt".format(epoch + 1, cfg.seed))
                )
                for key in train_history:
                    plot_history_v2(train_history[key], epoch, ckpt_dir, key + "_seed_{}.png".format(cfg.seed))

    if RANK == 0:
        torch.save(
            policy.module.state_dict(),
            os.path.join(ckpt_dir, "policy_last.ckpt")
        )

        sys.stdout.close()
        sys.stdout = original_stdout
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action = 'store', type = str, help = 'config file', required = True)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    cfg = edict(cfg)
    print(cfg)
    train(cfg)
