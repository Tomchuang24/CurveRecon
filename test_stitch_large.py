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
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
from sklearn.neighbors import KDTree, BallTree

def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class NormalizeUnitSphere(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def normalize(pcl, center=None, scale=None):
        """
        Args:
            pcl:  The point cloud to be normalized, (N, 3)
        """
        if center is None:
            p_max = pcl.max(dim=0, keepdim=True)[0]
            p_min = pcl.min(dim=0, keepdim=True)[0]
            center = (p_max + p_min) / 2    # (1, 3)
        pcl = pcl - center
        if scale is None:
            scale = (pcl ** 2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]  # (1, 1)
        pcl = pcl / scale
        return pcl, center, scale

    def __call__(self, data):
        assert 'pcl_noisy' not in data, 'Point clouds must be normalized before applying noise perturbation.'
        data['pcl_clean'], center, scale = self.normalize(data['pcl_clean'])
        data['center'] = center
        data['scale'] = scale
        return data



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

    data_path = "/home/tianz/project/CurveRecon/data/paris/"+args.fname

    # # # Point cloud
    # pcl = np.loadtxt(data_path)[:, 0:3]
    # pcl = torch.FloatTensor(pcl).to(device)

    # pcl = pcl.cpu().numpy()
    # # Build KDTree and extract leaves
    # leaf_size = 1024
    # bt = KDTree(pcl, leaf_size=leaf_size)
    # btData = bt.get_arrays()
    # btIndexes, btNodes = btData[1], btData[2]

    # # Identify leaf nodes
    # numNodes = btNodes.shape[0]
    # nodeIndex = np.arange(numNodes)
    # leaf_mask = np.array(btNodes['is_leaf'], dtype=bool)
    # leaf_nodes = nodeIndex[leaf_mask]

    # # Gather per-leaf point indices
    # points_of_node = [
    #     btIndexes[btNodes[t]['idx_start']:btNodes[t]['idx_end']]
    #     for t in leaf_nodes
    # ]

    # # Allocate prediction output
    # full_pred = np.zeros(pcl.shape[0], dtype=np.float32)

    # # Predict per leaf
    # for node_pts_idx in tqdm(points_of_node, desc='Predict Leaves'):
    #     pcl_part_noisy = torch.FloatTensor(pcl[node_pts_idx]).to(device)
    #     pcl_part_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_part_noisy)

    #     pred = model(pcl_part_noisy.unsqueeze(0)).squeeze(0).squeeze(-1)  # (N,)
    #     pred_rescaled = (pred * scale).detach().cpu().numpy()

    #     full_pred[node_pts_idx] = pred_rescaled  # assign prediction




    # Save to .npz file
    # np.savez_compressed(
    #     "./Results/"+ args.fname +".npz",
    #     points=pcl,           # original point coordinates
    #     prediction=full_pred  # per-point prediction (e.g., distance or heat)
    # )

    # Load and prepare point cloud
    pcl = np.loadtxt(data_path)[:, 0:3]
    pcl_tensor = torch.FloatTensor(pcl).to(device)
    pcl_np = pcl_tensor.cpu().numpy()
    num_points = pcl_np.shape[0]

    # Initialize min prediction per point as +inf
    full_pred_min = np.ones(num_points, dtype=np.float32) * np.inf

    # Sweep leaf sizes from 1024 to 1200 (inclusive) with step 40
    for leaf_size in range(1024, 1201, 20):
        print(f"Processing with leaf_size = {leaf_size}")

        # Build KDTree
        bt = KDTree(pcl_np, leaf_size=leaf_size)
        btIndexes, btNodes = bt.get_arrays()[1], bt.get_arrays()[2]

        # Identify leaf nodes
        leaf_mask = np.array(btNodes['is_leaf'], dtype=bool)
        leaf_nodes = np.arange(btNodes.shape[0])[leaf_mask]

        # Gather per-leaf point indices
        points_of_node = [
            btIndexes[btNodes[t]['idx_start']:btNodes[t]['idx_end']]
            for t in leaf_nodes
        ]

        # Predict per leaf and update minimum
        for node_pts_idx in tqdm(points_of_node, desc=f'Predict Leaves (leaf_size={leaf_size})'):
            pcl_part = torch.FloatTensor(pcl_np[node_pts_idx]).to(pcl_tensor.device)
            pcl_part, center, scale = NormalizeUnitSphere.normalize(pcl_part)

            pred = model(pcl_part.unsqueeze(0)).squeeze(0).squeeze(-1)  # (N,)
            pred_rescaled = (pred * scale).detach().cpu().numpy()

            # Update minimum prediction per point
            full_pred_min[node_pts_idx] = np.minimum(full_pred_min[node_pts_idx], pred_rescaled)

    # Replace inf for unseen points with 0.0
    full_pred_min[np.isinf(full_pred_min)] = 0.0


    np.savez_compressed(
        "./Results/"+ args.fname +".npz",
        points=pcl_np,           # original point coordinates
        prediction=full_pred_min  # per-point prediction (e.g., distance or heat)
    )
