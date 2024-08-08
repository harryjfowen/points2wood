# xxx

xxx

### 1. xxx


### Installation instructions to run on HPC

#torch cuda sanity checks
#torch.distributed.is_nccl_available()
#torch.version.cuda

1. module load cuda/11.4
2. python -m venv torch
3. source torch/bin/activate
4. pip install --upgrade pip
4. pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
5. CUDA="cu113" && URL="https://data.pyg.org/whl/torch" && VERSION="1.12.1"
6. pip install --no-index torch-scatter -f ${URL}-${VERSION}+${CUDA}.html
7. pip install --no-index torch-cluster -f ${URL}-${VERSION}+${CUDA}.html
7. pip install torch-sparse -f ${URL}-${VERSION}+${CUDA}.html
8. pip install torch-geometric



module load cuda/11.4
module load python/3.8
virtualenv venv 
source venv/bin/activate

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
CUDA="cu113" && URL="https://data.pyg.org/whl/torch" && VERSION="1.12.1"
pip install --no-index torch-scatter -f ${URL}-${VERSION}+${CUDA}.html
pip install torch-cluster -f ${URL}-${VERSION}+${CUDA}.html
pip install torch-sparse -f ${URL}-${VERSION}+${CUDA}.html
pip install torch-geometric