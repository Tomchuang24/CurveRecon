import os
import torch
import numpy as np
import random
import time
import logging
import logging.handlers

THOUSAND = 1000
MILLION = 1000000


class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self


class CheckpointManager(object):

    def __init__(self, save_dir, logger=BlackHole()):
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.ckpts = []
        self.logger = logger

        for f in os.listdir(self.save_dir):
            if f[:4] != 'ckpt':
                continue
            _, score, it = f.split('_')
            it = it.split('.')[0]
            self.ckpts.append({
                'score': float(score),
                'file': f,
                'iteration': int(it),
            })

    def get_worst_ckpt_idx(self):
        idx = -1
        worst = float('-inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] >= worst:
                idx = i
                worst = ckpt['score']
        return idx if idx >= 0 else None

    def get_best_ckpt_idx(self):
        idx = -1
        best = float('inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] <= best:
                idx = i
                best = ckpt['score']
        return idx if idx >= 0 else None
        
    def get_latest_ckpt_idx(self):
        idx = -1
        latest_it = -1
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['iteration'] > latest_it:
                idx = i
                latest_it = ckpt['iteration']
        return idx if idx >= 0 else None

    def save(self, model, args, score, others=None, step=None):

        if step is None:
            fname = 'ckpt_%.6f_.pt' % float(score)
        else:
            fname = 'ckpt_%.6f_%d.pt' % (float(score), int(step))
        path = os.path.join(self.save_dir, fname)

        torch.save({
            'args': args,
            'state_dict': model.state_dict(),
            'others': others
        }, path)

        self.ckpts.append({
            'score': score,
            'file': fname
        })

        return True

    def load_best(self):
        idx = self.get_best_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']))
        return ckpt
    
    def load_latest(self):
        idx = self.get_latest_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']))
        return ckpt

    def load_selected(self, file):
        ckpt = torch.load(os.path.join(self.save_dir, file))
        return ckpt


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', postfix='', prefix=''):
    log_dir = os.path.join(root, prefix + time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()) + postfix)
    os.makedirs(log_dir)
    return log_dir


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def int_list(argstr):
    return list(map(int, argstr.split(',')))


def str_list(argstr):
    return list(argstr.split(','))


def log_hyperparams(writer, log_dir, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k:v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {"hp_metric": -1})
    fw = writer._get_file_writer()
    fw.add_summary(exp)
    fw.add_summary(ssi)
    fw.add_summary(sei)
    with open(os.path.join(log_dir, 'hparams.csv'), 'w') as csvf:
        csvf.write('key,value\n')
        for k, v in vars_args.items():
            csvf.write('%s,%s\n' % (k, v))



def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def parse_experiment_name(name):
    if 'blensor' in name:
        if 'Ours' in name:
            dataset, method, tag, blensor_, noise = name.split('_')[:5]
        else:
            dataset, method, blensor_, noise = name.split('_')[:4]
        return {
            'dataset': dataset,
            'method': method,
            'resolution': 'blensor',
            'noise': noise,
        }

    if 'real' in name:
        if 'Ours' in name:
            dataset, method, tag, blensor_, noise = name.split('_')[:5]
        else:
            dataset, method, blensor_, noise = name.split('_')[:4]
        return {
            'dataset': dataset,
            'method': method,
            'resolution': 'real',
            'noise': noise,
        }
        
    else:
        if 'Ours' in name:
            dataset, method, tag, num_pnts, sample_method, noise = name.split('_')[:6]
        else:
            dataset, method, num_pnts, sample_method, noise = name.split('_')[:5]
        return {
            'dataset': dataset,
            'method': method,
            'resolution': num_pnts + '_' + sample_method,
            'noise': noise,
        }


import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict
from itertools import combinations

def laplacian_contraction(points, k=10, alpha=0.5, iterations=20):
    tree = cKDTree(points)
    new_points = points.copy()

    for _ in range(iterations):
        neighbors = tree.query(new_points, k=k + 1)[1][:, 1:]  # exclude self
        neighbor_means = np.mean(new_points[neighbors], axis=1)
        new_points = (1 - alpha) * new_points + alpha * neighbor_means

    return new_points

def smooth_heat(points, pred_heat, k=10, alpha=0.5, iterations=1):
    """
    Smooths the pred_heat values over the point cloud using KNN averaging.
    
    Args:
        points: (N, 3) array of point positions.
        pred_heat: (N,) array of heat values to smooth.
        k: number of neighbors.
        alpha: smoothing factor (0 = no smoothing, 1 = full neighbor average).
        iterations: number of smoothing passes.
        
    Returns:
        Smoothed pred_heat array.
    """
    tree = cKDTree(points)
    smoothed = pred_heat.copy()

    for _ in range(iterations):
        idx = tree.query(points, k=k + 1)[1][:, 1:]  # exclude self
        neighbor_heats = pred_heat[idx]
        neighbor_mean = neighbor_heats.mean(axis=1)
        smoothed = (1 - alpha) * smoothed + alpha * neighbor_mean
        pred_heat = smoothed.copy()  # for next iteration

    return smoothed

def build_edge_graph_efficient(foreground_points, original_points, k=10):
    # Build KDTree on original points
    tree = cKDTree(original_points)

    # Map: original point index → list of foreground point indices connected to it
    reverse_map = defaultdict(list)

    # For each foreground point, find kNN in original cloud
    neighbor_indices = tree.query(foreground_points, k=k)[1]

    for fg_idx, orig_neighbors in enumerate(neighbor_indices):
        for orig_idx in orig_neighbors:
            reverse_map[orig_idx].append(fg_idx)

    # Create edge set from shared original neighbors
    edge_set = set()
    for fg_indices in reverse_map.values():
        if len(fg_indices) > 1:
            for i, j in combinations(fg_indices, 2):
                edge = tuple(sorted((i, j)))
                edge_set.add(edge)

    return np.array(list(edge_set))

def map_to_closest_original(contracted_points, original_points):
    tree = cKDTree(original_points)
    closest_indices = tree.query(contracted_points)[1]
    return original_points[closest_indices], closest_indices