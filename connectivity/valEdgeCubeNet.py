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
from model.fieldnet import FeatFieldNet, EdgeCubeNet
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

def visualize_voxel_comparison(voxel_indices, pred_probs, gt_probs, log_dir, threshold=0.5, title="Voxel Occupancy Comparison", cnt = 0):
    """
    Show predicted vs ground-truth voxel occupancy in the same plot.
    - Red: predicted occupied
    - Green: ground-truth occupied
    - Yellow: correct prediction (intersection)
    """
    pred_mask = (pred_probs > threshold)
    gt_mask = (gt_probs > threshold)

    pred_coords = voxel_indices[pred_mask].cpu().numpy()
    gt_coords = voxel_indices[gt_mask].cpu().numpy()
    
    # Intersection (both pred and GT say occupied)
    intersection_mask = pred_mask & gt_mask
    intersection_coords = voxel_indices[intersection_mask].cpu().numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(gt_coords) > 0:
        ax.scatter(gt_coords[:, 0], gt_coords[:, 1], gt_coords[:, 2], c='green', marker='o', label='GT Only', alpha=0.5)
    if len(pred_coords) > 0:
        ax.scatter(pred_coords[:, 0], pred_coords[:, 1], pred_coords[:, 2], c='red', marker='^', label='Pred Only', alpha=0.5)
    if len(intersection_coords) > 0:
        ax.scatter(intersection_coords[:, 0], intersection_coords[:, 1], intersection_coords[:, 2], c='yellow', marker='s', label='Intersection')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(log_dir, "train_occupancy_prediction" + str(cnt) +".png")
    plt.savefig(plot_path)

    return


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
    model = EdgeCubeNet(args).to(device)

    log_dir = Path("/home/tianz/project/CurveRecon/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = os.path.join(log_dir, "EdgeCubeNet.pth")
   
    # Load into EdgeCubeNet's feature_net
    model.load_state_dict(torch.load(best_model_path))
 

    test_transforms = standard_train_transforms(noise_std_max=args.val_noise, 
                                                 noise_std_min=args.val_noise, 
                                                 rotate=False, 
                                                 scale_d=0)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)
  

    val_dataset = ConnectivityDataset(
            split='val',
            transform=test_transforms,
            overfit_mode=False,
            overfit_batch = 8,
            use_fps_knn=True,
            fps_K=200,
            knn_K=args.batch_points
            )
    
    val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=connectivity_collate_fn  # <-- this is crucial
    )

    

    val_loss_log = []
    val_MIOU_log = []
   
    best_val_loss = float('inf')

    
    val_loss_total = 0.0
    val_epoch_miou = 0.0

    res_log_dir = Path("/home/tianz/project/CurveRecon/logs/miou/")
    res_log_dir.mkdir(parents=True, exist_ok=True)

    cnt = 0

    with torch.no_grad():
        model.eval()
        # Evaluation
        print("Running validation...")
        
        for data in val_loader:
            cnt+=1
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
            }
            
            cube_occupancy = model(model_input)

            # Predicted occupancy
            pred_occupancy = cube_occupancy['pc_cube']  # (B*P, M)
            # Get ground truth cubes: shape (B, k, k, k)
            cubes = data['cubes'].to(device)  # (B, k, k, k)

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
            gt_occupancy = cubes[cube_batch_ids, x, y, z].to(device)                                  # (B*P, M)

            # Apply mask to ignore padded cube ids
            gt_occupancy = (gt_occupancy * grouped_cube_mask).float()                               # (B*P, M)
            pred_occupancy = (pred_occupancy * grouped_cube_mask).float()                           # (B*P, M)

            # === Loss Computation (example with BCEWithLogitsLoss or MSE) ===
            loss = F.binary_cross_entropy(pred_occupancy, gt_occupancy)


            # Binarize predictions and ground truth
            pred_bin = (pred_occupancy > 0.5).float()  # (B*P, M)
            gt_bin = (gt_occupancy > 0.5).float()      # (B*P, M)

            # Intersection and union
            intersection = (pred_bin * gt_bin).sum(dim=1)    # (B*P,)
            union = ((pred_bin + gt_bin) > 0).float().sum(dim=1)  # (B*P,)

            # Avoid division by zero
            iou = torch.where(union > 0, intersection / union, torch.ones_like(union))  # (B*P,)
            miou = iou.mean().item()

            if miou>0.1:
                # Flatten all cube ids, predictions, and ground truths
                cube_ids_all = model_input['grouped_cube_ids'].reshape(-1, 3)      # (B*P*M, 3)
                pred_occ_all = pred_occupancy.reshape(-1)                          # (B*P*M,)
                gt_occ_all = gt_occupancy.reshape(-1)                              # (B*P*M,)
                mask_all = grouped_cube_mask.reshape(-1)                           # (B*P*M,)

                # Apply mask to filter valid voxels
                valid_mask = (mask_all > 0)
                cube_ids_all = cube_ids_all[valid_mask]
                pred_occ_all = pred_occ_all[valid_mask]
                gt_occ_all = gt_occ_all[valid_mask]

                # Flatten all grouped input points (for visualizing patch point cloud)
                grouped_points_all = model_input['grouped_points'].reshape(-1, 3)  # (B*P*K, 3)

                

                

                
                threshold=0.5
                pred_mask = (pred_occ_all > threshold)
                gt_mask = (gt_occ_all > threshold)

                pred_coords = cube_ids_all[pred_mask].cpu().numpy()
                gt_coords = cube_ids_all[gt_mask].cpu().numpy()
                grouped_points_all = grouped_points_all.cpu().numpy()

      
                np.savez(
                    os.path.join(res_log_dir, f"sample_{cnt}_data.npz"),
                    pred_coords=pred_coords,
                    gt_coords=gt_coords,
                    grouped_points=grouped_points_all
                )
            
                # visualize_voxel_comparison(
                #     cube_ids_all, pred_occ_all, gt_occ_all,
                #     res_log_dir,
                #     threshold=0.5,
                #     title=f"Global Voxel Occupancy (All Patches, Sample{cnt})",
                #     cnt = cnt
                # )

            print(f"[Sample | mIoU = {miou:.4f} ")


            val_MIOU_log.append(miou)

  
    plot = True
    if plot:
        plt.figure()
        plt.plot(val_MIOU_log, label='mIoU_test', color='orange')
        plt.xlabel("Samples")
        plt.ylabel("Loss")
        plt.title("Training mIoU Curve")
        plt.grid(True)
        plt.legend()

        # Save the figure
        plot_path = os.path.join(log_dir, "val_loss_occupancy_curve.png")
        plt.savefig(plot_path)
        plt.close()

