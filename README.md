# points2wood

Semantic segmentation of wood and leaf in high resolution TLS point clouds

### 1. instalation

All development was done on Ubuntu LTS 22.04 on an NVIDIA Quadro RTX 6000 24GB 

First, install the ubuntu nvidia driver. I used 535.183.06  along with CUDA version 12.2 

Its recommended to set up a conda environment and use mamba to install packages wihtin it.

conda create --name myenv python=3.10
conda install mamba -c conda-forge

Second install pytorch. 2.3.1+cu121 and pytorch geometric  2.5.3

Instructions for each OS can be found at https://pytorch.org/get-started/locally/ and 
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

### 2. Running points2wood

conda activate myenv (myenv being the environemnt name you created above)

cd ~/points2wood/points2wood/ 

python3 predict.py --point-cloud /x/x/cloud.ply --model f1-eu.pth --batch_size 10 --is-wood 0.55 --grid_size 2.0 4.0 --min_pts 2048 --max_pts 16384;

*NOTE Make sure the point cloud contains columns x y z as a minimum and  x y z reflectance if available to you. 

*NOTE If the data contains relfectance but you dont want to use it, add the --reflectance_off flag 
