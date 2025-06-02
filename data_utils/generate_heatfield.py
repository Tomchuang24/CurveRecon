import pickle
import os
import numpy as np
import potpourri3d as pp3d



def load_curve_pkl(pkl_path):
    """Load curve points and edges from a .pkl file"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    points = np.array(data['points'])
    edges = np.array(data['edges'], dtype=int)
    return points, edges


if __name__ == '__main__':
    root_path = "/home/tianz/project/CurveRecon/data/"
    file_path = os.path.join(root_path, 'NerVE64Dataset', 'all.txt')

    file_list = np.loadtxt(file_path, dtype=int)

    for count, idx in enumerate(file_list):
        if count % 100 == 0:
            print(f'Process: {idx}')

        fname = '%08d' % idx
        obj_path = f"/home/tianz/project/CurveRecon/data/NerVE64Dataset/{fname}/pc_obj.pkl"
        curve_pkl = f"/home/tianz/project/CurveRecon/data/NerVE64Dataset/{fname}/step_curve_no_offset.pkl"
        heat_pkl = f"/home/tianz/project/CurveRecon/data/NerVE64Dataset/{fname}/pc_heat.pkl"

        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"OBJ file not found: {obj_path}")
        if not os.path.exists(curve_pkl):
            raise FileNotFoundError(f"Curve PKL file not found: {curve_pkl}")
        if os.path.exists(heat_pkl):
            print("skip")
            continue

        with open(obj_path, 'rb') as f:
            data = pickle.load(f)
            points = np.array(data['pc']-data['stable_offset'])

        # Load curve network
        curve_points, curve_edges = load_curve_pkl(curve_pkl)
        
        data = np.vstack((points, curve_points))
        rindexes = np.arange(len(data))
        indexes = rindexes[len(points):]
        solver = pp3d.PointCloudHeatSolver(data)
        dists = solver.compute_distance_multisource(indexes)
        
        res = {
            'points': points, 
            'heat': dists
        }
        
        with open(heat_pkl, 'wb') as f:
            pickle.dump(res, f)
    print("Finish!")
  