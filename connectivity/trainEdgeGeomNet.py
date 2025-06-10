import os
import sys
import torch
import argparse
import pickle
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F


BASE_DIR = os.path.dirname("/home/tianz/project/CurveRecon/")
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from model.fieldnet import EdgeGeomNet
from data_utils.heat_field_dataset import ConnectivityDataset
from data_utils.transforms import standard_train_transforms
from utils.loss import histogram_loss_batched



def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pad_and_stack(tensors, pad_value=0):
    max_len = max(t.shape[0] for t in tensors)
    padded = []
    for t in tensors:
        pad_size = [max_len - t.shape[0]] + list(t.shape[1:])
        padded_tensor = F.pad(t, (0, 0) * (t.dim() - 1) + (0, pad_size[0]), value=pad_value)
        padded.append(padded_tensor)
    return torch.stack(padded), max_len


def connectivity_collate_fn(batch):
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        if key == 'grouped_cube_ids' or key == 'grouped_cube_mask':
            # batch: list of [fps_K, M_i, 3]
            flat_list = []
            for item in batch:
                b_item = item[key]  # (fps_K, M, 3)
                flat_list.extend(list(b_item))  # list of [M_i, 3]
            padded, _ = pad_and_stack(flat_list)  # (B * fps_K, M_max, 3)
            collated[key] = padded
        else:
            collated[key] = torch.stack([item[key] for item in batch])

    return collated




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--noise_min', type=float, default=0.001)
    parser.add_argument('--noise_max', type=float, default=0.002)
    parser.add_argument('--val_noise', type=float, default=0.020)
    parser.add_argument('--fname', type=str, default="00000145")
    parser.add_argument('--score_net_hidden_dim', type=int, default=128)
    parser.add_argument('--score_net_num_blocks', type=int, default=4)
    parser.add_argument('--batch_points', type=int, default=1024,
                        help="Number of points to sample per iteration")
    parser.add_argument('--aug_rotate', type=eval, default=True, choices=[True, False])
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--max_grad_norm', type=float, default=float("inf"))
    args = parser.parse_args()

    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeGeomNet(args).to(device)

    # Load the checkpoint
    previous_best_path = "/home/tianz/project/CurveRecon/logs/fieldnet.pth"
    state_dict = torch.load(previous_best_path, map_location='cpu')
    # Extract only the 'feature_net' keys (they should be prefixed as 'feature_net.')
    feature_net_state_dict = {
        k.replace("feature_net.", ""): v
        for k, v in state_dict.items()
        if k.startswith("feature_net.")
    }
    # Load into EdgeCubeNet's feature_net
    model.feature_net.load_state_dict(feature_net_state_dict)
    # Freeze the feature extractor
    for param in model.feature_net.parameters():
        param.requires_grad = False


    optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    # Transforms and dataset
    train_transforms = standard_train_transforms(
        noise_std_min=args.noise_min,
        noise_std_max=args.noise_max,
        rotate=False,
    )

    test_transforms = standard_train_transforms(noise_std_max=args.val_noise, 
                                                 noise_std_min=args.val_noise, 
                                                 rotate=False, 
                                                 scale_d=0)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)
    
    train_dataset = ConnectivityDataset(
            split='train',
            transform=train_transforms,
            overfit_mode=False,
            overfit_batch = 8,
            use_fps_knn=True,
            fps_K=20,
            knn_K=args.batch_points
            )
    
    train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=connectivity_collate_fn  # <-- this is crucial
    )


    val_dataset = ConnectivityDataset(
            split='val',
            transform=train_transforms,
            overfit_mode=False,
            overfit_batch = 8,
            use_fps_knn=True,
            fps_K=20,
            knn_K=args.batch_points
            )
    
    val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=connectivity_collate_fn  # <-- this is crucial
    )

    log_dir = Path("/home/tianz/project/CurveRecon/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    loss_log = []
    MIOU_Log = []
    val_loss_log = []

    best_model_path = os.path.join(log_dir, "EdgeGeomNet.pth")
    best_val_loss = float('inf')


    loss_fn = torch.nn.L1Loss()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_miou = 0.0
        epoch_orig_grad_norm = 0.0

        for data in train_loader:
            # xyz_batch: [B, fps_K, knn_K, 3]
            # heat_gt_batch: [B, fps_K, knn_K]
            B, P, K, _ = data['grouped_points'].shape
            _, M, _ = data['grouped_cube_ids'].shape

             

            # Flatten B * P into batch dimension
            model_input = {
                'grouped_points':        data['grouped_points'].view(B * P, K, 3).to(device),      # (B*P, K, 3)
                'grouped_pc_grid_idxs':  data['grouped_pc_grid_idxs'].view(B * P, K, 3).to(device),# (B*P, K, 3)
                'grouped_cube_ids':      data['grouped_cube_ids'].view(B * P, M, 3).to(device),    # (B*P, M, 3)
                'grouped_cube_mask':     data['grouped_cube_mask'].view(B * P, M).to(device),      # (B*P, M)
                'cubes':                 data['cubes'].to(device),                          # (B, k, k, k, 3)
            }
            
            optimizer.zero_grad()
            cube_geom = model(model_input)

            # Predicted occupancy
            pred_geom = cube_geom['points']  # (B*P, M, 3)
            # Get ground truth cubes: shape (B, k, k, k)
            cubes = data['cubes'].to(device)  # (B, k, k, k)
            cube_points = data['cube_points'].to(device) # (B, k, k, k)

            # Flatten and move everything to device
            grouped_cube_ids = model_input['grouped_cube_ids']     # (B*P, M, 3)
            grouped_cube_mask = model_input['grouped_cube_mask']      # (B*P, M)
            # Map each patch to its original sample index in the batch
            cube_batch_ids = (torch.arange(B * P, device=device) // P).unsqueeze(1)       # (B*P, 1), broadcastable
            # Extract voxel indices
            x = grouped_cube_ids[..., 0]  # (B*P, M)
            y = grouped_cube_ids[..., 1]
            z = grouped_cube_ids[..., 2]
            # Index ground truth occupancy
            gt_occupancy = cubes[cube_batch_ids, x, y, z]                               # (B*P, M)
            gt_geom      = cube_points[cube_batch_ids, x, y, z]                         # (B*P, M, 3)
            # Apply mask to ignore padded cube ids
            gt_occupancy = (gt_occupancy * grouped_cube_mask).float()                               # (B*P, M)
            gt_occupancy_mask = (gt_occupancy > 0.5)

            # Select predicted points at ground truth positive voxels
            pred_geom_at_gt = pred_geom[gt_occupancy_mask]  
            gt_geom_valid   = gt_geom[gt_occupancy_mask]
            
  
            loss = loss_fn(pred_geom_at_gt, gt_geom_valid)
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_orig_grad_norm += orig_grad_norm.item()

            

        
        epoch_loss /= len(train_loader)
        epoch_orig_grad_norm /= len(train_loader)
        loss_log.append(epoch_loss)  # Save loss for plotting
        

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1:03d}] Loss = {epoch_loss:.6f}  |  Gradient Norm = {epoch_orig_grad_norm:.6f}")

        if (epoch + 1) % 25 == 0:
            scheduler.step()
            print(f"Learning rate decayed to {scheduler.get_last_lr()[0]:.6e}")


        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            # Evaluation
            print("Running validation...")
            
            for data in val_loader:
            # xyz_batch: [B, fps_K, knn_K, 3]
            # heat_gt_batch: [B, fps_K, knn_K]
                # xyz_batch: [B, fps_K, knn_K, 3]
            # heat_gt_batch: [B, fps_K, knn_K]
                B, P, K, _ = data['grouped_points'].shape
                _, M, _ = data['grouped_cube_ids'].shape

                

                # Flatten B * P into batch dimension
                model_input = {
                    'grouped_points':        data['grouped_points'].view(B * P, K, 3).to(device),      # (B*P, K, 3)
                    'grouped_pc_grid_idxs':  data['grouped_pc_grid_idxs'].view(B * P, K, 3).to(device),# (B*P, K, 3)
                    'grouped_cube_ids':      data['grouped_cube_ids'].view(B * P, M, 3).to(device),    # (B*P, M, 3)
                    'grouped_cube_mask':     data['grouped_cube_mask'].view(B * P, M).to(device),      # (B*P, M)
                    'cubes':                 data['cubes'].to(device),                          # (B, k, k, k, 3)
                }
                
                cube_geom = model(model_input)

                # Predicted occupancy
                pred_geom = cube_geom['points']  # (B*P, M, 3)
                # Get ground truth cubes: shape (B, k, k, k)
                cubes = data['cubes'].to(device)  # (B, k, k, k)
                cube_points = data['cube_points'].to(device) # (B, k, k, k)

                # Flatten and move everything to device
                grouped_cube_ids = model_input['grouped_cube_ids']     # (B*P, M, 3)
                grouped_cube_mask = model_input['grouped_cube_mask']      # (B*P, M)
                # Map each patch to its original sample index in the batch
                cube_batch_ids = (torch.arange(B * P, device=device) // P).unsqueeze(1)       # (B*P, 1), broadcastable
                # Extract voxel indices
                x = grouped_cube_ids[..., 0]  # (B*P, M)
                y = grouped_cube_ids[..., 1]
                z = grouped_cube_ids[..., 2]
                # Index ground truth occupancy
                gt_occupancy = cubes[cube_batch_ids, x, y, z]                               # (B*P, M)
                gt_geom      = cube_points[cube_batch_ids, x, y, z]                         # (B*P, M, 3)
                # Apply mask to ignore padded cube ids
                gt_occupancy = (gt_occupancy * grouped_cube_mask).float()                               # (B*P, M)
                gt_occupancy_mask = (gt_occupancy > 0.5)

                # Select predicted points at ground truth positive voxels
                pred_geom_at_gt = pred_geom[gt_occupancy_mask]  
                gt_geom_valid   = gt_geom[gt_occupancy_mask]
                
    
                loss = loss_fn(pred_geom_at_gt, gt_geom_valid)

                val_loss_total += loss.item()

        val_loss_avg = val_loss_total / len(val_loader)

        val_loss_log.append(val_loss_avg)
        print(f"[Epoch {epoch+1:03d}] Val Loss = {val_loss_avg:.6f}")
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)
            print(f"✅ Saved new best model with loss {best_val_loss:.6f}")
                

    plot = True
    if plot:
        plt.figure()
        plt.plot(loss_log, label='Train Loss', color='blue')
        plt.plot(val_loss_log, label='Validation Loss', color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training loss Curve of mid-point of curve-edges")
        plt.grid(True)
        plt.legend()

        # Save the figure
        plot_path = os.path.join(log_dir, "train_loss_mid-point_curve.png")
        plt.savefig(plot_path)
        plt.close()

