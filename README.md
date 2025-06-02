
# Learning General Distance to Feature Field for Shape Modelling

**Authors:** Tian Zhang, Sagi Filin  
**Affiliation:** Technion – Israel Institute of Technology

---
![Teaser](Teasor/teasor.png)
## 🧠 Overview

This project introduces a learning-based framework for predicting **general Distance to Feature Fields (DFF)** from 3D point clouds. Instead of detecting discrete feature points or curves, we learn a **continuous field** that encodes proximity to geometric features (e.g., edges and corners).

Our method enables:
- Sharp feature detection
- Curve reconstruction from dense fields
- Geometry-aware shape understanding
- Robust modeling across varying scales and shapes

---

## 🔍 Key Highlights

- 📌 **Patch-based heat field learning** using FPS + KNN sampling  
- 🔁 **Min-based stitching** to aggregate overlapping patch predictions  
- 📐 **Loss functions**: Histogram-based + L1 loss for fine-grained distance regression  
- 🧩 **Feature curve reconstruction** through voxel graph and spline fitting  
- 🔬 **Polyscope visualization** for interactive inspection

---

## 📁 Repository Structure

```
DistanceFeatureField/
├── models/                # PyTorch model definitions
├── datasets/              # Custom dataset and patch sampling logic
├── losses/                # Loss functions including histogram loss
├── visualization/         # Polyscope utilities for heat field display
├── train.py               # Training script
├── test.py                # Evaluation and stitching logic
├── data/                  # Preprocessed dataset (e.g., NerVE64)
└── Results/               # Output predictions and logs
```

---

## ⚙️ Setup

```bash
git clone https://github.com/yourusername/DistanceFeatureField.git
cd DistanceFeatureField
pip install -r requirements.txt
```

Dependencies include:
- PyTorch
- Open3D
- Polyscope
- NumPy, SciPy, tqdm

---

## 📊 Dataset

We use a processed subset of the [ABC Dataset](https://deep-geometry.github.io/abc-dataset/).  
Ground truth is computed via OpenCascade and saved in `.pkl` files under `data/NerVE64Dataset/`.

---

## 🚀 Training

```bash
python train.py --batch_points 1024 --fps_K 20
```

Trains the model using patch-based sampling and geometry-aware loss.

---

## 🧪 Testing and Visualization

```bash
python test.py --eval --save_vis
```

Evaluates predictions and stitches them using min aggregation.  
Polyscope can be used to visualize the output heat maps on full shapes.

---

## 🖼️ Visualization Example

```python
import polyscope as ps
ps.init()
ps.register_point_cloud("points", xyz_np, radius=0.002)
ps.get_point_cloud("points").add_scalar_quantity("heat", heat_np, enabled=True)
ps.show()
```

---

## 📄 Citation

If you find this project useful, please cite:

```bibtex
@article{zhang2025dff,
  title={Learning General Distance to Feature Field for Shape Modelling},
  author={Zhang, Tian and Filin, Sagi},
  journal={arXiv preprint arXiv:2506.XXXX},
  year={2025}
}
```

---

## 📬 Contact

**Tian Zhang**  
📧 tianzhang [at] campus.technion.ac.il  
🌐 [Technion – Israel Institute of Technology](https://www.technion.ac.il/)
