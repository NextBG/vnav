import os
import io
import time
import wandb
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

def train_eval_loop_vnav(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    optimizer: AdamW,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    action_stats: Dict[str, float],
    log_folder: str,
    epochs: int,
    start_epoch: int,
    vis_interval: int,
    prob_mask: float,
    goal_norm_factor: float,
    use_wandb: bool,
    device: torch.device,
):
    # Normalize
    img_tf = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    action_stats = (torch.tensor(action_stats["x_max"], device=device), torch.tensor(action_stats["y_abs_max"], device=device))

    '''
        Train eval loop
    '''
    for epoch in range(start_epoch, start_epoch+epochs):
        # Log
        print(f">>> Epoch {epoch}/{start_epoch+epochs-1} <<<")
        if use_wandb:
            wandb.log({
                "Train/Learning rate": lr_scheduler.get_last_lr()[0],
                "Train/Epoch": epoch,
            })

        '''
            Train
        '''
        # Log variables
        last_time = time.time()
        batch_total_loss = 0.0
        
        # Train loop
        model.train()
        for _, data in tqdm(
            enumerate(train_dataloader), 
            desc="Training batches", 
            total=len(train_dataloader), 
            bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            ):
            # Load data
            (obs_imgs, goal_vec, actions, _, _) = data
            obs_imgs = obs_imgs.to(device)  # [B, N, 3, H, W]
            goal_vec = goal_vec.to(device)  # [B, 2]
            actions = actions.to(device)    # [B, H, 2]

            # Batch size
            BS = obs_imgs.shape[0]

            # Actions
            actions = F.pad(actions, (0, 0, 1, 0), mode="constant", value=0)    # [B, H, 2] -> [B, H+1, 2]
            deltas = torch.diff(actions, dim=1)                                 # [B, H, 2]
            n_deltas = normalize(deltas, action_stats)                          # [B, H, 2]

            # Add noise
            noise = torch.randn(n_deltas.shape, device=device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,(BS,), device=device)
            noisy_action = noise_scheduler.add_noise(n_deltas, noise, timesteps)

            # Normalize images
            n_obs_imgs = img_tf(obs_imgs)

            # Normalize goal
            dists = torch.norm(goal_vec, dim=1)             # [B]
            n_dists = torch.tanh(dists * goal_norm_factor)  # [B]
            n_coeff = (n_dists/dists).unsqueeze(1)          # [B, 1]
            n_goal_vec = goal_vec * n_coeff                 # [B, 2]

            # Goal mask
            goal_mask = (torch.rand((BS, ), device=device) < prob_mask).int()

            # Encode context
            context_token = model(
                "vision_encoder", 
                obs_imgs=n_obs_imgs,
                goal_vec=n_goal_vec,
                goal_mask=goal_mask,
            )

            # Predict noise
            noise_pred = model(
                "noise_predictor", 
                sample=noisy_action, 
                timestep=timesteps, 
                global_cond=context_token
            )

            # Diffusion loss
            diffusion_loss = F.mse_loss(noise_pred, noise)
            batch_total_loss += diffusion_loss.item()

            # Backward
            optimizer.zero_grad()
            diffusion_loss.backward()
            optimizer.step()

            # log
            if use_wandb:
                wandb.log({"Train/Diffusion loss": diffusion_loss.item()})
                wandb.log({"Train/Speed [it per s]": 1/(time.time()-last_time)})
                last_time = time.time()

        # Batch log
        batch_avg_loss = batch_total_loss / len(train_dataloader)
        if use_wandb:
            wandb.log({"Train/Batch average loss": batch_avg_loss})

        # Step scheduler
        lr_scheduler.step()

        # Save paths
        numbered_path = os.path.join(log_folder, f"{epoch}.pth")
        latest_path = os.path.join(log_folder, f"latest.pth")
        
        # Save
        torch.save(model.state_dict(), numbered_path)
        torch.save(model.state_dict(), latest_path)

        # Save metadata
        metadata = {"current_epoch": start_epoch+epoch+1,}
        with open(os.path.join(log_folder, f"metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
            
        '''
            Evaluation
        '''
        # Init
        model.eval()

        # Log
        batch_total_loss = 0.0

        with torch.no_grad():
            for i, data in tqdm(
                enumerate(eval_dataloader), 
                desc="Evaluation batches", 
                total=len(eval_dataloader),
                bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                ):
                # Load data
                (obs_imgs, goal_vec, actions, yaws, metadata) = data
                obs_imgs = obs_imgs.to(device)  # [B, N, 3, H, W]
                goal_vec = goal_vec.to(device)  # [B, 2]
                actions = actions.to(device)    # [B, H, 2]

                # Batch size
                BS = obs_imgs.shape[0]

                # Normalize images
                n_obs_imgs = img_tf(obs_imgs)

                # Normalize goal
                dists = torch.norm(goal_vec, dim=1)             # [B]
                n_dists = torch.tanh(dists * goal_norm_factor)  # [B]
                n_coeff = (n_dists/dists).unsqueeze(1)          # [B, 1]
                n_goal_vec = goal_vec * n_coeff                 # [B, 2]

                # Normalize action
                pad_actions = F.pad(actions, (0, 0, 1, 0), mode="constant", value=0)    # [B, H, 2] -> [B, H+1, 2]
                deltas = torch.diff(pad_actions, dim=1)                                 # [B, H, 2]
                n_deltas = normalize(deltas, action_stats)                              # [B, H, 2]

                # Add noise 
                noise = torch.randn(n_deltas.shape, device=device)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (BS,), device=device)
                noisy_deltas = noise_scheduler.add_noise(n_deltas, noise, timesteps)

                # Goal mask
                goal_mask = (torch.rand((BS, ), device=device) < prob_mask).int()

                # Encode context
                context_token = model(
                    "vision_encoder", 
                    obs_imgs=n_obs_imgs,
                    goal_vec=n_goal_vec,
                    goal_mask=goal_mask,
                )
                
                # Predict noise
                noise_pred = model(
                    "noise_predictor", 
                    sample=noisy_deltas, 
                    timestep=timesteps, 
                    global_cond=context_token
                )
                
                # Diffusion loss
                diffusion_loss = F.mse_loss(noise_pred, noise)
                batch_total_loss += diffusion_loss.item()

                # Log
                if use_wandb:
                    wandb.log({"Evaluation/Diffusion loss": diffusion_loss.item()})

                    # Visualize
                    if i % vis_interval == 0:
                        # Sample actions
                        sampled_actions = sample_actions(
                            model=model,
                            noise_scheduler=noise_scheduler,
                            n_obs_imgs=n_obs_imgs[0],
                            n_goal_vec=n_goal_vec[0],
                            goal_mask=goal_mask[0],
                            num_samples=10,
                            pred_horizon=len(n_deltas[0]),
                            action_stats=action_stats,
                            device=device,
                        )

                        visualize(
                            obs_imgs=obs_imgs[0],
                            gt_actions=actions[0],
                            gt_yaws=yaws[0],
                            goal_vec=goal_vec[0],
                            goal_mask=goal_mask[0],
                            metadata=metadata[0],
                            sampled_actions=sampled_actions,
                        )

        # Batch log
        batch_avg_loss = batch_total_loss / len(eval_dataloader)
        if use_wandb:
            wandb.log({"Evaluation/Batch average loss": batch_avg_loss})

def sample_actions(
        model: nn.Module,
        noise_scheduler: DDPMScheduler,
        n_obs_imgs: torch.Tensor,
        n_goal_vec: torch.Tensor,
        goal_mask: torch.Tensor,
        num_samples: int,
        pred_horizon: int,
        action_stats: torch.Tensor,
        device: torch.device,
):
    # Encode context
    context_token = model(
        func_name="vision_encoder", 
        obs_imgs=n_obs_imgs.unsqueeze(0), 
        goal_vec=n_goal_vec.unsqueeze(0),
        goal_mask=goal_mask.unsqueeze(0),
    ) # [1, E]
    context_token = context_token.repeat(num_samples, 1) # [N, E]

    # Sample noise
    diffusion_out = torch.randn((num_samples, pred_horizon, 2), device=device) # [N, H, 2]

    # Denoise
    for i in noise_scheduler.timesteps:
        # Predict noise
        noise_pred = model(
            func_name="noise_predictor",
            sample=diffusion_out,
            timestep=i.repeat(num_samples).to(device),
            global_cond=context_token,
        ) # [N, H, 2]

        # Remove noise
        diffusion_out = noise_scheduler.step(
            model_output=noise_pred,
            timestep=i,
            sample=diffusion_out,
        ).prev_sample # [N, H, 2]

    # Unnormalize
    deltas = unnormalize(diffusion_out, action_stats) # [N, H, 2]

    # Get actions
    actions = torch.cumsum(deltas, dim=1) 

    return actions

def visualize(
        obs_imgs: torch.Tensor,
        gt_actions: torch.Tensor,
        gt_yaws: torch.Tensor,
        goal_vec: torch.Tensor,
        goal_mask: torch.Tensor,
        metadata: Dict[str, str],
        sampled_actions: torch.Tensor,
        ):
    '''
        Plot observations
    '''
    # Params
    context_size = obs_imgs.shape[0]
    num_samples = sampled_actions.shape[0]
    dataset_name, traj_name, traj_idx, timestep = metadata.values()

    # Init figure
    fig = plt.figure(figsize=(10 * (context_size-1), 10))
    fig.suptitle(f"Dataset: {dataset_name}, Trajectory: {traj_name}_{traj_idx}, Timestep: {timestep}")

    # Detach
    obs_imgs = obs_imgs.detach().cpu().numpy()                  # [N, 3, H, W]
    sampled_actions = sampled_actions.detach().cpu().numpy()    # [S, H, 2]
    gt_actions = gt_actions.detach().cpu().numpy()              # [H, 2]
    gt_yaws = gt_yaws.detach().cpu().numpy()                    # [H]
    goal_vec = goal_vec.detach().cpu().numpy()                  # [2]
    goal_mask = goal_mask.detach().cpu().numpy()                # [1]

    # # Plot observations
    obs_imgs = np.transpose(obs_imgs, (0, 2, 3, 1)) # [N, 3, H, W] -> [N, H, W, 3]
    # for i in range(context_size-1):
    #     ax = fig.add_subplot(1, context_size-1, i+1)
    #     ax.set_title(f"Observation T-{context_size-1-i}")
    #     ax.imshow(obs_imgs[i]) 

    # # Log
    # wandb.log({"Evaluation/Observations": wandb.Image(fig)})
    # plt.close(fig)

    '''
        Plot last obs and trajectories (ROS convention)
    '''
    # Init figure
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"Dataset: {dataset_name}, Trajectory: {traj_name}_{traj_idx}, Timestep: {timestep}")

    # Last observation
    ax = fig.add_subplot(121)
    ax.set_title("Observation T")
    ax.imshow(obs_imgs[-1])
    
    # Trajectory
    ax = fig.add_subplot(122)
    ax.grid()
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    ax.axis("equal")

    # Plot ground truth actions
    gt_actions = np.concatenate([np.zeros((1,2)), gt_actions], axis=0)                          # [H+1, 2]
    gt_yaws = np.concatenate([np.zeros((1,)), gt_yaws], axis=0)                                 # [H+1]
    sampled_actions = np.concatenate([np.zeros((num_samples, 1, 2)), sampled_actions], axis=1)  # [S, H+1, 2]

    # Plot sampled trajectories
    for i in range(sampled_actions.shape[0]):
        x = sampled_actions[i, :, 0]
        y = sampled_actions[i, :, 1]
        ax.plot(-y, x, "r-", alpha=0.4, linewidth=2.0)
        ax.plot(-y, x, "ro", alpha=0.4, markersize=4)

    # Ground truth
    gt_x = gt_actions[:, 0]
    gt_y = gt_actions[:, 1]
    ax.plot(-gt_y, gt_x, "g-o", markersize=3) # GT trajectory

    # Goal
    if goal_mask == 0:
        ax.plot(-goal_vec[1], goal_vec[0], "gx", markersize=0.01)
    else:
        ax.plot(-goal_vec[1], goal_vec[0], "bo", markersize=30)
        # ax.quiver(0, 
        #           0, 
        #           -goal_vec[1], 
        #           goal_vec[0], 
        #           angles='xy', 
        #           scale_units='xy', 
        #           scale=1,
        #           color='g',
        #           alpha=0.5,
        #           width=0.01,
        #           headwidth=6,
        #           label="Goal vector"
        #           )

    # Yaw
    ax.quiver(
        -gt_y,
        gt_x,
        -np.sin(gt_yaws),
        np.cos(gt_yaws),
        color='g', 
        alpha=0.5,
        width=0.005,
    ) # GT yaws

    # Log
    wandb.log({"Evaluation/Actions": wandb.Image(fig)})
    plt.close(fig)

def normalize(data: torch.Tensor, stats: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    x_max, y_abs_max = stats

    # Normalize x [0, ACTION_MAX_X] -> [-1, 1]
    data[:, :, 0] = data[:, :, 0] / x_max * 2 - 1

    # Normalize y [-ACTION_MAX_Y, ACTION_MAX_Y] -> [-1, 1]
    data[:, :, 1] = data[:, :, 1] / y_abs_max

    return data

def unnormalize(ndata: torch.Tensor, stats: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    x_max, y_abs_max = stats

    # Unnormalize x [-1, 1] -> [0, ACTION_MAX_X]
    ndata[:, :, 0] = (ndata[:, :, 0] + 1) / 2 * x_max

    # Unnormalize y [-1, 1] -> [-ACTION_MAX_Y, ACTION_MAX_Y]
    ndata[:, :, 1] = ndata[:, :, 1] * y_abs_max

    return ndata