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


            # grouped_points = torch.stack([points_sub[idxs] for idxs in gather_idx])  # (fps_K, knn_K, 3)
            # grouped_heat = torch.stack([heat_sub[idxs] for idxs in gather_idx])      # (fps_K, knn_K)

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



def custom_collate_fn(batch):
    # batch: list of dicts
    collated = {}
    keys = batch[0].keys()
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor) and all(v.shape == values[0].shape for v in values):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values  # keep as list
    return collated


class ConnectivityDataset(Dataset):
    def __init__(self, split='train',
                 root_path='/home/tianz/project/CurveRecon/data/NerVE64Dataset',
                 transform=None,
                 overfit_mode=False,
                 overfit_batch = 10,
                 use_fps_knn=True,
                 fps_K=20,
                 knn_K=1024):
        super().__init__()
        self.root_path = root_path
        self.transform = transform
        self.overfit_mode = overfit_mode

        # FPS + KNN config
        self.use_fps_knn = use_fps_knn
        self.fps_K = fps_K
        self.knn_K = knn_K

        list_path = os.path.join(root_path, f'{split}.txt')
        file_list = np.loadtxt(list_path, dtype=int)
        file_names = [f'{idx:08d}' for idx in file_list]

        self.valid_file_names = []
        for fname in file_names:
            pkl_path = os.path.join(root_path, fname, 'pc_heat.pkl')
            obj_path = os.path.join(root_path, fname, 'pc_obj.pkl')
            nerve_path = os.path.join(root_path, fname, 'nerve_reso64.pkl')
            if os.path.exists(pkl_path) and os.path.exists(obj_path) and os.path.exists(nerve_path):
                self.valid_file_names.append(fname)
       
        if self.overfit_mode:
            self.overfit_batch = overfit_batch
            self.valid_file_names = self.valid_file_names[:self.overfit_batch ]
        print(f'[Dataloader] {split} set length:', len(self.valid_file_names))

            
    def generate_cube_shifted_ID(self, pcid, k):
        device = pcid.device

        grid = torch.zeros((k, k, k), dtype=torch.bool, device=device)
        grid[pcid[:, 0], pcid[:, 1], pcid[:, 2]] = True

        shifts = torch.tensor([[i, j, l] for i in range(-1, 2)
                                        for j in range(-1, 2)
                                        for l in range(-1, 2)],
                            device=device)

        expanded = pcid.unsqueeze(1) + shifts.unsqueeze(0)  # (N, 27, 3)
        expanded = expanded.view(-1, 3)
        mask = ((expanded >= 0) & (expanded < k)).all(dim=1)
        expanded = expanded[mask]
        expanded = torch.unique(expanded, dim=0)
        return expanded  # (M, 3)

    def __len__(self):
        return len(self.valid_file_names)

    def __getitem__(self, idx):
        fname = self.valid_file_names[idx]
        folder = os.path.join(self.root_path, fname)

        # Load point-level data
        with open(os.path.join(folder, 'pc_heat.pkl'), 'rb') as f:
            heat_data = pickle.load(f)
        with open(os.path.join(folder, 'pc_obj.pkl'), 'rb') as f:
            pc_data = pickle.load(f)
        with open(os.path.join(folder, 'nerve_reso64.pkl'), 'rb') as f:
            nerve_data = pickle.load(f)

        heat = heat_data['heat'].astype(np.float32)      # (N,)
        
        #
        points = pc_data['pc'].astype(np.float32)  # (N,3) here coordinates with offsets
        # this needed to be generated per patch
        pc_grid_idxs = pc_data['pc_grid_idx']      # (N,3) index of voxels where its neighbors are non-empty
        
        #connectivity related data
        edge_grid_idx = nerve_data['cube_idx']
        k = nerve_data['grid_size']
        eg = edge_grid_idx
        cubes = np.zeros((k,k,k), dtype=bool)
        faces = np.zeros((k,k,k,3), dtype=bool)
        
        cubes[eg[:,0], eg[:,1], eg[:,2]] = True
        faces[eg[:,0], eg[:,1], eg[:,2]] = nerve_data['cube_faces']
        
        # grids where the feature passed
        cpoints = nerve_data['cube_points']
        step = 2. / k
        # transform points to local cube coordinates
        centers = step*edge_grid_idx + (step/2. - 1.)
        cpoints -= centers
        cpoints *= k # ocuupied cube center

        cube_points= np.zeros((k,k,k,3), dtype=bool)
        cube_points[eg[:,0], eg[:,1], eg[:,2]] = cpoints


        if not self.use_fps_knn:
            # Default random sampling
            N = points.shape[0]
            if N >= self.num_points:
                indices = np.random.choice(N, self.num_points, replace=False)
            else:
                indices = np.random.choice(N, self.num_points, replace=True)

            points = points[indices]
            heat = heat[indices]
            pc_grid_idxs = pc_grid_idxs[indices]
   
            if self.transform is not None:
                points, heat = self.transform(points, heat)

            return torch.from_numpy(points).float(), torch.from_numpy(heat).float(), torch.from_numpy(pc_grid_idxs).long() 
        
        else:
            # FPS + KNN sampling
            # if points.shape[0] < self.sample_size:
            #     raise ValueError(f"Point cloud too small: {points.shape[0]} < {self.sample_size}")
            points_sub = torch.from_numpy(points)  # (sample_size, 3)
            heat_sub = torch.from_numpy(heat)      # (sample_size,)
            pc_grid_idxs_sub = torch.from_numpy(pc_grid_idxs)      # (sample_size,)

            fps_K = self.fps_K  # or define the number of centers directly
            center_idx = torch.randperm(points_sub.shape[0])[:fps_K]  # randomly permute and pick first K
            centers = points_sub[center_idx]  # (fps_K, 3)
    
            # Step 3: KNN to gather patches
            # Step 1: ensure points_sub is (1, N, 3)
            knn = knn_points(centers.unsqueeze(0), points_sub.unsqueeze(0), K=self.knn_K)  # (1, 20, 1024)
            gather_idx = knn.idx.squeeze(0)  # (20, 1024)
       
            grouped_points = points_sub[gather_idx]  # (20, 1024, 3)
            grouped_heat = heat_sub[gather_idx]      # (20, 1024)
            grouped_pc_grid_idxs = pc_grid_idxs_sub[gather_idx] # (20, 1024, 3)
        

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

            grouped_cube_ids = []
            cube_valid_masks = []

            

            max_cubes = 0
            for i in range(fps_K):
                patch_grid_idx = grouped_pc_grid_idxs[i].to(torch.long)  # (knn_K, 3)
                cids = self.generate_cube_shifted_ID(patch_grid_idx, k)  # (M_i, 3)
                grouped_cube_ids.append(cids)
                cube_valid_masks.append(torch.ones(cids.shape[0], dtype=torch.bool))
                max_cubes = max(max_cubes, cids.shape[0])

            # Pad all grouped_cube_ids to (max_cubes, 3)
            grouped_cube_ids_padded = torch.zeros((fps_K, max_cubes, 3), dtype=torch.long)
            cube_valid_masks_padded = torch.zeros((fps_K, max_cubes), dtype=torch.bool)

            for i in range(fps_K):
                M_i = grouped_cube_ids[i].shape[0]
                grouped_cube_ids_padded[i, :M_i] = grouped_cube_ids[i]
                cube_valid_masks_padded[i, :M_i] = cube_valid_masks[i]

            return {
                'grouped_points': grouped_points.float(),                    # (fps_K, knn_K, 3)
                'grouped_heat': grouped_heat.float(),                        # (fps_K, knn_K)
                'grouped_pc_grid_idxs': grouped_pc_grid_idxs.long(),         # (fps_K, knn_K, 3)
                'grouped_cube_ids': grouped_cube_ids_padded,                 # (fps_K, max_M, 3)
                'grouped_cube_mask': cube_valid_masks_padded,                # (fps_K, max_M)
                'cubes': torch.from_numpy(cubes),                            # (k, k, k)
                'faces': torch.from_numpy(faces),                            # (k, k, k, 3)
                'cube_points': torch.from_numpy(cube_points).float()         # (k, k, k, 3)
            }


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