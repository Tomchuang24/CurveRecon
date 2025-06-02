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


BASE_DIR = os.path.dirname("/home/tianz/project/CurveRecon/")
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from model.fieldnet import FeatFieldNet
from data_utils.heat_field_dataset import HeatFieldFullPatchTestDataset
from data_utils.transforms import standard_train_transforms
from utils.loss import histogram_loss_batched



def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--noise_min', type=float, default=0.005)
    parser.add_argument('--noise_max', type=float, default=0.015)
    parser.add_argument('--val_noise', type=float, default=0.005)
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
    model = FeatFieldNet(args).to(device)
    log_dir = Path("/home/tianz/project/CurveRecon/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = os.path.join(log_dir, "fieldnet.pth")

    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    test_dataset = HeatFieldFullPatchTestDataset(
    split='test',
    fps_K=50,
    knn_K=args.batch_points
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    loss_log = []
    val_loss_log = []
    plot = True
    
    
    model.eval()
    val_loss_total = 0.0
    best_val_loss = float('inf')
    
    Res_dir = Path("/home/tianz/project/CurveRecon/Results")
    Res_dir.mkdir(parents=True, exist_ok=True)

    val_loss_total = 0.0
    save_limit = 10  # only save the first 10 results
    save_count = 0

    with torch.no_grad():
        for batch in test_loader:
            points = batch['full_points'][0].to(device)            # (N, 3)
            heat_gt = batch['full_heat'][0].to(device)             # (N,)
            patches = batch['patch_points'][0].to(device)          # (num_patches, knn_K, 3)
            patch_indices = batch['patch_indices'][0]              # (num_patches, knn_K)
            fname = batch['fname'][0]

            N = points.shape[0]
            pred_heat_all = torch.full((N,), float('inf'), device=device)  # Use min aggregation

            for p, idx in zip(patches, patch_indices):
                center = p.mean(dim=0, keepdim=True)                     # (1, 3)
                p_centered = p - center
                scale = torch.norm(p_centered, dim=1).max() + 1e-8       # avoid divide by zero
                p_normalized = p_centered / scale                        # (knn_K, 3)

                pred = model(p_normalized.unsqueeze(0)).squeeze(0).squeeze(-1)  # (knn_K,)
                pred_rescaled = pred * scale                             # undo scale (heat ∝ dist)

                pred_heat_all[idx] = torch.min(pred_heat_all[idx], pred_rescaled)

            loss_hist = histogram_loss_batched(pred_heat_all.unsqueeze(0), heat_gt.unsqueeze(0))
            loss_l1 = torch.nn.functional.l1_loss(pred_heat_all, heat_gt)
            loss = loss_hist + loss_l1
            val_loss_total += loss.item()

            print(f"[{fname}] Histogram Loss: {loss_hist:.6f}, L1 Loss: {loss_l1:.6f}")

            # Save prediction result
            if save_count < save_limit:
                out_path = Res_dir / f"{fname}_pred.npz"
                np.savez_compressed(
                    out_path,
                    points=points.cpu().numpy(),
                    pred_heat=pred_heat_all.cpu().numpy(),
                    gt_heat=heat_gt.cpu().numpy()
                )
                save_count += 1

    val_loss_avg = val_loss_total / len(test_loader)
    print(f"Overall Val Loss = {val_loss_avg:.6f}")

