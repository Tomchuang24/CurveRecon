import torch
from torch import nn
import pytorch3d.ops
import numpy as np

from .feature import FeatureExtraction
from .score import ScoreNet
from itertools import product
from torch_scatter import scatter_mean

def get_random_indices(n, m):
    assert m < n
    return np.random.permutation(n)[:m]


class FeatFieldNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
                # networks
        self.feature_net = FeatureExtraction()
        self.score_net = ScoreNet(
            z_dim=self.feature_net.out_channels,
            out_dim=1,
            hidden_size=args.score_net_hidden_dim,
            num_blocks=args.score_net_num_blocks,
        )

    def forward(self, pcl_noisy):
        """
        Denoising score matching.
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
        """
        B, N_noisy = pcl_noisy.size(0), pcl_noisy.size(1)

        # Feature extraction
        feat = self.feature_net(pcl_noisy)  # (B, N, F)
        distance = self.score_net(feat)
        
        return distance #, target, scores, noise_vecs



class EdgeCubeNet(nn.Module):
    def __init__(self, args):
        super(EdgeCubeNet, self).__init__()
        self.args = args
        self.predict_type = 'BCE'
        self.k = 64  # Grid resolution

        # Feature extraction and score prediction
        self.feature_net = FeatureExtraction()

        self.conv3d = nn.Sequential(
        nn.Conv3d(self.feature_net.out_channels, self.feature_net.out_channels//2, kernel_size=3, padding=1),
        #nn.BatchNorm3d(self.feature_net.out_channels),
        nn.ReLU(inplace=True)
        )
      
        self.score_net = ScoreNet(
            z_dim=self.feature_net.out_channels//2,
            out_dim=1,
            hidden_size=args.score_net_hidden_dim,
            num_blocks=args.score_net_num_blocks,
        )

        if self.predict_type == 'BCE':
            self.activation = nn.Sigmoid()
        elif self.predict_type == 'NLL':
            self.activation = nn.LogSoftmax(dim=-1)
        else:
            self.activation = nn.Softmax(dim=-1)

    def forward(self, model_input):
        # === Inputs ===
        points    = model_input['grouped_points']         # (B, N, 3)
        pcid      = model_input['grouped_pc_grid_idxs']   # (B, N, 3)
        cube_ids  = model_input['grouped_cube_ids']       # (B, M, 3)
       
        k = self.k
        B, N, _ = points.shape

        # === Feature Extraction ===
        feat = self.feature_net(points)                   # (B, N, C)
        C = feat.shape[-1]

        # === Efficient Grid Aggregation ===
        feat_flat = feat.view(B * N, C)                   # (B*N, C)
        pcid_flat = pcid.view(B * N, 3)                   # (B*N, 3)
        batch_idx = torch.arange(B, device=points.device).view(B, 1).expand(B, N).reshape(-1)  # (B*N,)
        flat_idx = batch_idx * k**3 + pcid_flat[:, 0]*k*k + pcid_flat[:, 1]*k + pcid_flat[:, 2]  # (B*N,)

        f_voxel_flat = scatter_mean(feat_flat, flat_idx.long(), dim=0, dim_size=B * k**3)  # (B*k^3, C)
        f_voxels = f_voxel_flat.view(B, k, k, k, C).permute(0, 4, 1, 2, 3).contiguous()  # (B, C, k, k, k)
        f_voxels = self.conv3d(f_voxels)  # (B, C, k, k, k)
        # === Extract Cube Features ===
        b_idx = torch.arange(B, device=points.device).view(B, 1, 1)                          # (B, 1, 1)
        b_idx = b_idx.expand(-1, cube_ids.shape[1], 1)                                       # (B, M, 1)
        cube_idx = torch.cat([b_idx, cube_ids], dim=2)                                       # (B, M, 4)

        # Unpack indices
        b, x, y, z = cube_idx.unbind(dim=2)                                                  # each (B, M)
        cube_feats = f_voxels[b, :, x, y, z]                                                 # (B, M, C)

        # === Score Prediction ===
        cube_logits = self.score_net(cube_feats)                                             # (B, M, 1)
        cube_occupancy = self.activation(cube_logits).squeeze(-1)                            # (B, M)

        return {
            'pc_cube': cube_occupancy,  # (B, M)
        }


class EdgeGeomNet(nn.Module):
    def __init__(self, args):
        super(EdgeGeomNet, self).__init__()
        self.args = args
        self.k = 64  # Grid resolution

        # Feature extraction and score prediction
        self.feature_net = FeatureExtraction()

        self.conv3d = nn.Sequential(
        nn.Conv3d(self.feature_net.out_channels, self.feature_net.out_channels, kernel_size=3, padding=1),
        #nn.BatchNorm3d(self.feature_net.out_channels),
        nn.ReLU(inplace=True)
        )
      
        self.score_net = ScoreNet(
            z_dim=self.feature_net.out_channels,
            out_dim=3,
            hidden_size=args.score_net_hidden_dim,
            num_blocks=args.score_net_num_blocks,
        )


    def forward(self, model_input):
        points    = model_input['grouped_points']         # (B, N, 3)
        pcid      = model_input['grouped_pc_grid_idxs']   # (B, N, 3)
        cube_ids  = model_input['grouped_cube_ids']       # (B, M, 3)
   
        B, N, _ = points.shape
        k = self.k

        # === Feature Extraction ===
        feat = self.feature_net(points)  # (B, N, C)
        C = feat.shape[-1]

        # === Grid Aggregation ===
        feat_flat = feat.view(B * N, C)
        pcid_flat = pcid.view(B * N, 3)
        batch_idx = torch.arange(B, device=points.device).view(B, 1).expand(B, N).reshape(-1)
        flat_idx = batch_idx * k**3 + pcid_flat[:, 0]*k*k + pcid_flat[:, 1]*k + pcid_flat[:, 2]
        f_voxel_flat = scatter_mean(feat_flat, flat_idx.long(), dim=0, dim_size=B * k**3)
        f_voxels = f_voxel_flat.view(B, k, k, k, C).permute(0, 4, 1, 2, 3).contiguous()
        f_voxels = self.conv3d(f_voxels)

        # === Cube Feature Extraction ===
        b_idx = torch.arange(B, device=points.device).view(B, 1, 1).expand(-1, cube_ids.shape[1], 1)
        cube_idx = torch.cat([b_idx, cube_ids], dim=2)
        b, x, y, z = cube_idx.unbind(dim=2)
        cube_feats = f_voxels[b, :, x, y, z]  # (B, M, C)

        # === Score Prediction ===
        out_point = self.score_net(cube_feats)  # (B, M, 3)
        out_point = torch.clamp(out_point, min=-1., max=1.)
        return {
            'points': out_point,
        }




class EdgeFaceNet(nn.Module):
    def __init__(self, args):
        super(EdgeFaceNet, self).__init__()
        self.args = args
        self.k = 64  # Grid resolution
        self.predict_type = 'BCE'

        # Feature extraction and score prediction
        self.feature_net = FeatureExtraction()

        self.conv3d = nn.Sequential(
        nn.Conv3d(self.feature_net.out_channels, self.feature_net.out_channels//4, kernel_size=3, padding=1),
        #nn.BatchNorm3d(self.feature_net.out_channels),
        nn.ReLU(inplace=True)
        )
      
        self.score_net = ScoreNet(
            z_dim=(self.feature_net.out_channels//4) *4,
            out_dim=3,
            hidden_size=args.score_net_hidden_dim,
            num_blocks=args.score_net_num_blocks,
        )

        if self.predict_type == 'BCE':
            self.activation = nn.Sigmoid()

        self.cube_neighbor = torch.LongTensor([
                [-1, 0, 0], [0, -1, 0], [0, 0, -1]
            ])
        
    def forward(self, model_input):
        points    = model_input['grouped_points']         # (B, N, 3)
        pcid      = model_input['grouped_pc_grid_idxs']   # (B, N, 3)
        cube_ids  = model_input['grouped_cube_ids']       # (B, M, 3)
   
        B, N, _ = points.shape
        k = self.k

        # === Feature Extraction ===
        feat = self.feature_net(points)  # (B, N, C)
        C = feat.shape[-1]

        # === Grid Aggregation ===
        feat_flat = feat.view(B * N, C)
        pcid_flat = pcid.view(B * N, 3)
        batch_idx = torch.arange(B, device=points.device).view(B, 1).expand(B, N).reshape(-1)
        flat_idx = batch_idx * k**3 + pcid_flat[:, 0]*k*k + pcid_flat[:, 1]*k + pcid_flat[:, 2]
        f_voxel_flat = scatter_mean(feat_flat, flat_idx.long(), dim=0, dim_size=B * k**3)
        f_voxels = f_voxel_flat.view(B, k, k, k, C).permute(0, 4, 1, 2, 3).contiguous()
        f_voxels = self.conv3d(f_voxels)

        # === Cube Feature Extraction ===
        M = cube_ids.shape[1]
        b_idx = torch.arange(B, device=points.device).view(B, 1, 1).expand(-1, M, 1)
        cube_idx = torch.cat([b_idx, cube_ids], dim=2)
        b, x, y, z = cube_idx.unbind(dim=2)
        # neighbors: (3, 3)
        neighbor_shifts = self.cube_neighbor.to(cube_ids.device)  # (3, 3)
        # Expand cube_ids: (B, M, 1, 3)
        cube_ids_exp = cube_ids.unsqueeze(2).expand(-1, -1, 3, -1)
        # Apply shifts: (B, M, 3, 3)
        neighbor_ids = cube_ids_exp + neighbor_shifts.view(1, 1, 3, 3)
        # Clamp neighbor indices to grid bounds
        neighbor_ids = neighbor_ids.clamp(0, k - 1)  # still (B, M, 3, 3)
        # Unpack indices
        x_n = neighbor_ids[..., 0]  # (B, M, 3)
        y_n = neighbor_ids[..., 1]
        z_n = neighbor_ids[..., 2]
        # Expand batch index: (B, M, 3)
        b_idx = torch.arange(B, device=cube_ids.device).view(B, 1, 1).expand(B, M, 3)
        # Index into voxel grid: f_voxels[b, :, x, y, z] with advanced indexing
        neighbor_feats = f_voxels[b_idx, :, x_n, y_n, z_n]  # (B, M, 3, C)
        # Get center features: (B, M, C)
        center_feats = f_voxels[b, :, x, y, z]
        # Concatenate center and neighbor features: (B, M, 4*C)
        all_feats = torch.cat([center_feats.unsqueeze(2), neighbor_feats], dim=2)  # (B, M, 4, C)
        all_feats = all_feats.view(B, M, -1)  # Flatten last two dims → (B, M, 4*C)

        
        # === Score Prediction ===
        out_face = self.score_net(all_feats)  # (B, M, 3)
       
        return {
            'pc_face': self.activation(out_face),
        }


