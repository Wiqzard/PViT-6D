

## Get Startet

### Install Pytorch3d
pytorch3d https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

#### Or follow:
conda create -n pytorch3d python=3.9 \\
conda activate pytorch3d \\
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html


### Install Requirements 
pip install timm
pip install opencv-python
pip install matplotlib
pip install psutil 
pip install scipy 
pip install dill 
pip install trimesh 
pip install imgaug 


### Setup 
- Setup config inf configs/direct.yaml with your paths 
- Configure parameters in run.py for train/val