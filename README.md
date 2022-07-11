# Symmetry-Aware Transformer-based Mirror Detection -- SATNet
This repo is the official implementation of ["Symmetry-Aware Transformer-based Mirror Detection"]()

## Installation

Our project is based on [MMsegmentation](https://github.com/open-mmlab/mmsegmentation) and [Swin-Transformer-Semantic-Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation). Please follow the official [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) for installation and dataset preparation. We recommend to create a conda environment and install dependencies in Linux as follows:

```shell
conda create -n satnet python=3.7 -y
conda activate satnet

conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch -y
pip install mmcv-full==1.2.2 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
git clone https://github.com/tyhuang0428/SATNet
cd SATNet
pip install -e .  # or "python setup.py develop"
pip install -r requirements/optional.txt

mkdir data
```

## Data preparation
We train and evaluate our SATNet on Mirror Segmentation Dataset (MSD), Progressive Mirror Dataset (PMD), RGB-D Mirror Dataset (RGBD-Mirror). You can download the pack file [mirror.tar]() of these datasets and unpack it to `SATNet/data`:


## Get Started

### Pre-trained model
We provide our pre-trained checkpoints and corresponding training logs [here](https://drive.google.com/drive/folders/1R6Odc8mgSrk3WYGz1fK3tL-MBfr8dk0_?usp=sharing) for all the three datasets.

### Train

```
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}

# Config file of our SATNet is in the folder: ./configs/satnet/
# For example, train our SATNet on the MSD dataset with 8 GPUs
./tools/dist_train.sh ./configs/satnet/msd_satnet.py 8
```

* Tensorboard

  If you want to use tensorboard, you need to `pip install tensorboard` and uncomment the Line 6 `dict(type='TensorboardLoggerHook')` of `SETR/configs/_base_/default_runtime.py`.


### Single-scale testing

```
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} --eval mIoU

# For example, test our SATNet on MSD dataset with a single GPU
./tools/dist_test.sh ./configs/satnet/msd_satnet.py ckpts/msd/iter_20000.pth 1 --eval mIoU
```

Please see [getting_started.md](docs/getting_started.md) for the more basic usage of training and testing.

