# Symmetry-Aware Transformer-based Mirror Detection
This repo is the official implementation of AAAI 2023 paper ["Symmetry-Aware Transformer-based Mirror Detection"](https://arxiv.org/abs/2207.06332)

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
We train and evaluate our SATNet on Mirror Segmentation Dataset (MSD), Progressive Mirror Dataset (PMD), RGB-D Mirror Dataset (RGBD-Mirror). You can download zip files for corresponding datasets [here](https://drive.google.com/drive/folders/1Fj0fIwn-mXI3xTlENiHXjYNLMUBRTZwg?usp=sharing) and unpack them to `SATNet/data`

## Results and Models
| Dataset | IoU | F | MAE | model |
| :---: | :---: | :---: | :---: | :---: |
| MSD | 85.41 | 0.922 | 0.033 | [Google Drive](https://drive.google.com/file/d/1jWdcPUi-UDxhTeAFOtQxGxTmP3TRsrfd/view?usp=sharing) |
| PMD | 69.38 | 0.847 | 0.025 | [Google Drive](https://drive.google.com/file/d/1JGkVNqNjn13E6I_O_UAxM1DCCgMZ8FmP/view?usp=sharing) |
| RGBD-Mirror | 78.42 | 0.906 | 0.031 | [Google Drive](https://drive.google.com/file/d/1XqZiNzBgkJX9MxZlZ-QEozEVjRi0Z8xF/view?usp=sharing) |

Pre-trained Swin-S are available [here](https://drive.google.com/file/d/1syNvj0VUpN7WDSwrppc6f0KmMKbOeXkz/view?usp=sharing).

## Get Started
### Train

```
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} --load-from ${Swin-S_CHECKPOINT_FILE}

# Config file of our SATNet is in the folder: ./configs/satnet/
# For example, train our SATNet on the MSD dataset with 8 GPUs
./tools/dist_train.sh ./configs/satnet/msd_satnet.py 8 --load-from ./swin.pth
```

* Tensorboard

  If you want to use tensorboard, you need to `pip install tensorboard` and uncomment the Line 6 `dict(type='TensorboardLoggerHook')` of `SATNet/configs/_base_/default_runtime.py`.


### Testing

```
python ./tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU

# For example, test our SATNet on MSD dataset
python ./tools/test.py ./configs/satnet/msd_satnet.py ./msd.pth --eval mIoU
```

### Visualization
```
python ./tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --show-dir ${SAVE_PATH}

# For example, visulaize the results of our SATNet on MSD dataset
python ./tools/test.py ./configs/satnet/msd_satnet.py ./msd.pth --show --show-dir ./results/msd
```

Please see [getting_started.md](docs/getting_started.md) for the more basic usage of training and testing.
