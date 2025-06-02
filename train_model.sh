#!/bin/bash
source $HOME/miniconda3/etc/profile.d/conda.sh

#gdown --id 15t7lh1NhyZ1n1w8vhxQvJsFTRMkReK1U
conda activate score-denoise
cd /home/tianz/project/CurveRecon/
python train.py --epochs 100
python test.py


