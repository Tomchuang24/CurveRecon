import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch3d.ops import knn_points

class HeatFieldDataset(Dataset):
    def __init__(self, split='train',
                 root_path='/home/tianz/project/CurveRecon/data/NerVE64Dataset',
                 num_points=2048,
                 transform=None,
                 overfit_mode=False,
                 use_fps_knn=True,
                 sample_size=24000,
                 fps_K=20,
                 knn_K=1024):
        super().__init__()
        self.root_path = root_path
        self.num_points = num_points
        self.transform = transform
        self.overfit_mode = overfit_mode

        # FPS + KNN config
        self.use_fps_knn = use_fps_knn
        self.sample_size = sample_size
        self.fps_K = fps_K
        self.knn_K = knn_K

        list_path = os.path.join(root_path, f'{split}.txt')
        file_list = np.loadtxt(list_path, dtype=int)
        file_names = [f'{idx:08d}' for idx in file_list]

        self.valid_file_names = []
        for fname in file_names:
            pkl_path = os.path.join(root_path, fname, 'pc_heat.pkl')
            if os.path.exists(pkl_path):
                self.valid_file_names.append(fname)
        print(f'[Dataloader] {split} set length:', len(self.valid_file_names))

        if self.overfit_mode:
            self.valid_file_names = self.valid_file_names[:100]

    def __len__(self):
        return len(self.valid_file_names)

    def __getitem__(self, idx):
        fname = self.valid_file_names[idx]
        pkl_path = os.path.join(self.root_path, fname, 'pc_heat.pkl')

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        points = data['points'].astype(np.float32)  # (N, 3)
        heat = data['heat'].astype(np.float32)      # (N,)

        if not self.use_fps_knn:
            # Default random sampling
            N = points.shape[0]
            if N >= self.num_points:
                indices = np.random.choice(N, self.num_points, replace=False)
            else:
                indices = np.random.choice(N, self.num_points, replace=True)

            points = points[indices]
            heat = heat[indices]

            if self.transform is not None:
                points, heat = self.transform(points, heat)

            return torch.from_numpy(points).float(), torch.from_numpy(heat).float()

        else:
            # FPS + KNN sampling
            # if points.shape[0] < self.sample_size:
            #     raise ValueError(f"Point cloud too small: {points.shape[0]} < {self.sample_size}")

            points_sub = torch.from_numpy(points)  # (sample_size, 3)
            heat_sub = torch.from_numpy(heat)      # (sample_size,)

            fps_K = self.fps_K  # or define the number of centers directly
            center_idx = torch.randperm(points_sub.shape[0])[:fps_K]  # randomly permute and pick first K
            centers = points_sub[center_idx]  # (fps_K, 3)
    
            # Step 3: KNN to gather patches
            # Step 1: ensure points_sub is (1, N, 3)
            knn = knn_points(centers.unsqueeze(0), points_sub.unsqueeze(0), K=self.knn_K)  # (1, 20, 1024)
            gather_idx = knn.idx.squeeze(0)  # (20, 1024)
       
            grouped_points = points_sub[gather_idx]  # (20, 1024, 3)
            grouped_heat = heat_sub[gather_idx]      # (20, 1024)


            grouped_points = torch.stack([points_sub[idxs] for idxs in gather_idx])  # (fps_K, knn_K, 3)
            grouped_heat = torch.stack([heat_sub[idxs] for idxs in gather_idx])      # (fps_K, knn_K)

            if self.transform is not None:
                # Apply transform per patch
                pts_list, heat_list = [], []
                for i in range(grouped_points.shape[0]):
                    p_np, h_np = grouped_points[i].numpy(), grouped_heat[i].numpy()
                    p_tr, h_tr = self.transform(p_np, h_np)
                    pts_list.append(torch.from_numpy(p_tr))
                    heat_list.append(torch.from_numpy(h_tr))
                grouped_points = torch.stack(pts_list)
                grouped_heat = torch.stack(heat_list)

            return grouped_points.float(), grouped_heat.float()




class HeatFieldFullPatchTestDataset(Dataset):
    def __init__(self, 
                 root_path='/home/tianz/project/CurveRecon/data/NerVE64Dataset', 
                 split='test',
                 fps_K=20, 
                 knn_K=1024):
        super().__init__()
        self.root_path = root_path
        self.fps_K = fps_K
        self.knn_K = knn_K


        list_path = os.path.join(root_path, f'{split}.txt')
        file_list = np.loadtxt(list_path, dtype=int)
        self.file_names = [f'{idx:08d}' for idx in file_list]

        self.valid_file_names = []
        for fname in self.file_names:
            pkl_path = os.path.join(root_path, fname, 'pc_heat.pkl')
            if os.path.exists(pkl_path):
                self.valid_file_names.append(fname)
        print(f'[FullTestDataset] {split} set length:', len(self.valid_file_names))

    def __len__(self):
        return len(self.valid_file_names)

    def __getitem__(self, idx):
        fname = self.valid_file_names[idx]
        pkl_path = os.path.join(self.root_path, fname, 'pc_heat.pkl')

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # points = torch.from_numpy(data['points'].astype(np.float32))  # (N, 3)
        # heat = torch.from_numpy(data['heat'].astype(np.float32))      # (N,)

        # center_idx = torch.randperm(points.shape[0])[:self.fps_K]
        # centers = points[center_idx]

        # knn = knn_points(centers.unsqueeze(0), points.unsqueeze(0), K=self.knn_K)
        # gather_idx = knn.idx.squeeze(0)

        # patch_points = torch.stack([points[idxs] for idxs in gather_idx])
        # patch_indices = gather_idx  # To stitch later

        points = torch.from_numpy(data['points'].astype(np.float32))  # (N, 3)
        heat = torch.from_numpy(data['heat'].astype(np.float32))      # (N,)
        N = points.shape[0]

        # Normalize temporarily to speed up knn (optional, remove if using full resolution)
        points_mean = points.mean(dim=0)
        points_norm = points - points_mean

        # Create coverage mask
        covered = torch.zeros(N, dtype=torch.bool)
        patch_points = []
        patch_indices = []

        # Precompute KNN matrix once
        knn_all = knn_points(points_norm.unsqueeze(0), points_norm.unsqueeze(0), K=self.knn_K)
        knn_indices = knn_all.idx.squeeze(0)  # (N, knn_K)

        while not torch.all(covered):
            # Find first uncovered point
            center_idx = (~covered).nonzero(as_tuple=False)[0].item()
            knn_idx = knn_indices[center_idx]  # (knn_K,)

            patch_points.append(points[knn_idx])
            patch_indices.append(knn_idx)

            covered[knn_idx] = True

        patch_points = torch.stack(patch_points)     # (num_patches, knn_K, 3)
        patch_indices = torch.stack(patch_indices)   # (num_patches, knn_K)

        return {
            'full_points': points,
            'full_heat': heat[:len(points)],
            'patch_points': patch_points,
            'patch_indices': patch_indices,
            'fname': fname
        }