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
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--noise_min', type=float, default=0.005)
    parser.add_argument('--noise_max', type=float, default=0.015)
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
    model = FeatFieldNet(args).to(device)
    previous_best_path = "/home/tianz/project/CurveRecon/logs/fieldnet_final.pth"
    model.load_state_dict(torch.load(previous_best_path))
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    # Transforms and dataset
    train_transforms = standard_train_transforms(
        noise_std_min=args.noise_min,
        noise_std_max=args.noise_max,
        rotate=args.aug_rotate
    )
    test_transforms = standard_train_transforms(noise_std_max=args.val_noise, 
                                                 noise_std_min=args.val_noise, 
                                                 rotate=False, 
                                                 scale_d=0)
    
    train_dataset = HeatFieldDataset(
    split='train',
    transform=train_transforms,
    overfit_mode=False,
    use_fps_knn=True,
    fps_K=20,
    knn_K=args.batch_points
    )
    val_dataset = HeatFieldDataset(
        split='val',
        transform=test_transforms,
        overfit_mode=False,
        use_fps_knn=True,
        fps_K=20,
        knn_K=args.batch_points
    )

    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)

    log_dir = Path("/home/tianz/project/CurveRecon/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    loss_log = []
    val_loss_log = []

    plot = True
    best_val_loss = float('inf')

    best_model_path = os.path.join(log_dir, "fieldnet.pth")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_orig_grad_norm = 0.0
        for xyz_batch, heat_gt_batch in train_loader:
        # xyz_batch: [B, fps_K, knn_K, 3]
        # heat_gt_batch: [B, fps_K, knn_K]
            B, P, K, _ = xyz_batch.shape  # B=batch, P=patches, K=knn

            xyz_input = xyz_batch.view(B * P, K, 3).to(device)           # (B * P, K, 3)
            heat_gt = heat_gt_batch.view(B * P, K).to(device)            # (B * P, K)

            optimizer.zero_grad()
            heat_pred = model(xyz_input).squeeze(-1) # (B, N,)

            loss_hist = histogram_loss_batched(heat_pred, heat_gt)
            loss_l1 = torch.nn.functional.l1_loss(heat_pred, heat_gt)
            loss = loss_hist + loss_l1

            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_orig_grad_norm += orig_grad_norm.item()

        epoch_loss /= len(train_loader)
        epoch_orig_grad_norm /= len(train_loader)
        loss_log.append(epoch_loss)  # Save loss for plotting

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1:03d}] Loss = {epoch_loss:.6f} | Gradient Norm = {epoch_orig_grad_norm:.6f}")
        
        if (epoch + 1) % 25 == 0:
            scheduler.step()
            print(f"Learning rate decayed to {scheduler.get_last_lr()[0]:.6e}")


        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            # Evaluation
            print("Running validation...")
            for xyz_batch, heat_gt_batch in val_loader:
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
            
    if plot:
        plt.figure()
        plt.plot(loss_log, label='Train Loss', color='blue')
        plt.plot(val_loss_log, label='Validation Loss', color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.legend()

        # Save the figure
        plot_path = os.path.join(log_dir, "train_loss_curve.png")
        plt.savefig(plot_path)
        plt.close()

    