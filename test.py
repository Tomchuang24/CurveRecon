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
from data_utils.heat_field_dataset import HeatFieldDataset
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


    test_transforms = standard_train_transforms(noise_std_max=args.val_noise, 
                                                 noise_std_min=args.val_noise, 
                                                 rotate=False, 
                                                 scale_d=0)
    
    test_dataset = HeatFieldDataset(
    split='test',
    transform=test_transforms,
    overfit_mode=False,
    use_fps_knn=True,
    fps_K=20,
    knn_K=args.batch_points
    )
    

    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

    
    loss_log = []
    val_loss_log = []

    plot = True
    best_val_loss = float('inf')

   
    model.eval()
    val_loss_total = 0.0
    best_val_loss = float('inf')
    with torch.no_grad():
        # Evaluation
        print("Running validation...")
        for xyz_batch, heat_gt_batch in test_loader:
        # xyz_batch: [B, fps_K, knn_K, 3]
        # heat_gt_batch: [B, fps_K, knn_K]

            B, P, K, _ = xyz_batch.shape  # B=batch, P=patches, K=knn

            xyz_input = xyz_batch.view(B * P, K, 3).to(device)           # (B * P, K, 3)
            heat_gt = heat_gt_batch.view(B * P, K).to(device)            # (B * P, K)

            heat_pred = model(xyz_input).squeeze(-1)
            loss_hist = histogram_loss_batched(heat_pred, heat_gt)
            loss_l1 = torch.nn.functional.l1_loss(heat_pred, heat_gt)
            loss = loss_hist + loss_l1


            print(f"[Val Sample] Histogram Loss: {loss_hist:.6f}, L1 Loss: {loss_l1:.6f}")

            val_loss_total += loss.item()

    val_loss_avg = val_loss_total / len(test_loader)
    val_loss_log.append(val_loss_avg)
    print(f"Overall Val Loss = {val_loss_avg:.6f}")
    
    