#!/bin/bash
source $HOME/miniconda3/etc/profile.d/conda.sh

#gdown --id 15t7lh1NhyZ1n1w8vhxQvJsFTRMkReK1U
conda activate score-denoise
cd /home/tianz/project/CurveRecon/data_utils
python generate_heatfield.py


