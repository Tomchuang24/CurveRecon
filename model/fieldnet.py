import torch
from torch import nn
import pytorch3d.ops
import numpy as np

from .feature import FeatureExtraction
from .score import ScoreNet


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

    