import os
import time
import yaml
import wandb
import pickle
import argparse
import numpy as np

# Torch
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset

from warmup_scheduler import GradualWarmupScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Custiom imports
from utils import *
from model import Vnav
from vnav_dataset import VnavDataset, vnav_collect_fn
from train_eval_loop import train_eval_loop_vnav

# Main
def main(config):
    '''
        Initialization
    '''
    # Device
    device = torch.device(config["device"])

    # wandb
    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project= config["project_name"],
            config=config
            )
        wandb.run.name = config["run_name"]

    # Seed
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True

    # Cudnn
    torch.backends.cudnn.benchmark = True

    '''
        Datasets
    '''
    train_datasets = []
    eval_datasets = []
    for dataset_name in config["dataset_names"]:
        for dataset_type in ["train", "eval"]:
            dataset = VnavDataset(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                config=config,
            )
            # Train datasets
            if dataset_type == "train":
                train_datasets.append(dataset)
            # Test datasets
            elif dataset_type == "eval":
                eval_datasets.append(dataset)

    # Train dataloader
    train_dataset = ConcatDataset(train_datasets)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        collate_fn=vnav_collect_fn,
    )

    # Eval dataloader
    eval_dataset = ConcatDataset(eval_datasets)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        collate_fn=vnav_collect_fn,
    )
    
    '''
        Model
    '''
    # Model
    model = Vnav(
        enc_dim=config["encoding_dim"],
        context_size=config["context_size"],
        pred_horizon=config["pred_horizon"],
    ).to(device)

    # Update config
    if config["use_wandb"]:
        n_params = count_parameters(model, print_table=False)
        config["n_params"] = n_params
        wandb.config.update(config)
        print(f"Total Trainable Params: {n_params/1e6:.2f}M")

    # Noise Scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"], # Diffusion iterations
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=float(config["lr"])
    )

    # Scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config["epochs"]-config["warmup_epochs"],
    )

    # Warmup Scheduler
    lr_scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=config["warmup_epochs"],
        after_scheduler=lr_scheduler
    )

    # Load checkpoint
    current_epoch = 0
    if "checkpoint" in config:
        # Load checkpoint
        checkpoint_folder = os.path.join(config["logs_folder"], config["checkpoint"]) 
        latest_checkpoint = torch.load(os.path.join(checkpoint_folder, f"latest.pth"))
        model.load_state_dict(latest_checkpoint)

        # Load metadata
        with open(os.path.join(checkpoint_folder, f"metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
        current_epoch = metadata["current_epoch"]

        # Print
        print(f"Loading checkpoint from {checkpoint_folder}")
            
    '''
        Train
    '''
    # Train and eval loop
    train_eval_loop_vnav(
        model=model,
        train_dataloader=train_dataloader, 
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        noise_scheduler=noise_scheduler,
        action_stats=config["action_stats"],
        epochs=config["epochs"],
        log_folder=config["log_folder"],
        start_epoch=current_epoch,
        vis_interval=config["vis_interval"],
        prob_mask=config["prob_mask"],
        goal_norm_factor=config["goal_norm_factor"],
        use_wandb=config["use_wandb"],
        device=device,
    )

'''
    Main
'''
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", type=str, default="cuda:0")
    args = parser.parse_args()

    # Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Device
    config["device"] = args.device

    # Log folder
    config["run_name"] = "vnav-" + time.strftime("%y%m%d-%H%M%S")
    config["log_folder"] = os.path.join(config["logs_folder"], config["run_name"])
    os.makedirs(os.path.join(config["log_folder"], config["run_name"]))

    # Start
    print(f"Start at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Train and eval loop
    main(config)

    # Finish
    print(f"Finished at {time.strftime('%Y-%m-%d %H:%M:%S')} > <")