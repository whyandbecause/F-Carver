# F-Carver: Carving Features for Reformulating Representations in Domain Generalized Segmentation (Under Review in TMM)
  
## Environment Setup
To set up your environment, execute the following commands:
```bash
conda create -n F-Carver -y
conda activate F-Carver
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
pip install "mmdet>=3.0.0"
pip install xformers=='0.0.20' # optional for DINOv2
pip install -r requirements.txt
pip install future tensorboard
```

## Dataset Preparation
The Preparation is similar as [DDB](https://github.com/xiaoachen98/DDB).

**Cityscapes:** Download `leftImg8bit_trainvaltest.zip` and `gt_trainvaltest.zip` from [Cityscapes Dataset](https://www.cityscapes-dataset.com/downloads/) and extract them to `data/cityscapes`.

**Mapillary:** Download MAPILLARY v1.2 from [Mapillary Research](https://research.mapillary.com/) and extract it to `data/mapillary`.

**GTA:** Download all image and label packages from [TU Darmstadt](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract them to `data/gta`.

Prepare datasets with these commands:
```shell
cd F-Carver
mkdir data
# Convert data for validation if preparing for the first time
python tools/convert_datasets/gta.py data/gta # Source domain
python tools/convert_datasets/cityscapes.py data/cityscapes
# Convert Mapillary to Cityscapes format and resize for validation
python tools/convert_datasets/mapillary2cityscape.py data/mapillary data/mapillary/cityscapes_trainIdLabel --train_id
python tools/convert_datasets/mapillary_resize.py data/mapillary/validation/images data/mapillary/cityscapes_trainIdLabel/val/label data/mapillary/half/val_img data/mapillary/half/val_label
```
(Optional) **ACDC**: Download all image and label packages from [ACDC](https://acdc.vision.ee.ethz.ch/) and extract them to `data/acdc`.

The final folder structure should look like this:

```
F-Carver
├── ...
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── bdd100k
│   │   ├── images
│   │   |   ├── 10k
│   │   │   |    ├── train
│   │   │   |    ├── val
│   │   ├── labels
│   │   |   ├── sem_seg
│   │   |   |    ├── masks
│   │   │   |    |    ├── train
│   │   │   |    |    ├── val
│   ├── mapillary
│   │   ├── training
│   │   ├── cityscapes_trainIdLabel
│   │   ├── half
│   │   │   ├── val_img
│   │   │   ├── val_label
│   ├── gta
│   │   ├── images
│   │   ├── labels
├── ...
```
## Pretraining Weights
* **Download:** Download pre-trained weights from [facebookresearch](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) for testing. Place them in the project directory without changing the file name.
* **Convert:** Convert pre-trained weights for training or evaluation.
  ```bash
  python tools/convert_models/convert_dinov2.py checkpoints/dinov2_vitl14_pretrain.pth checkpoints/dinov2_converted.pth
  ```
  (optional for 1024x1024 resolution)
  ```bash
  python tools/convert_models/convert_dinov2.py checkpoints/dinov2_vitl14_pretrain.pth checkpoints/dinov2_converted_1024x1024.pth --height 1024 --width 1024
  ```
## Training
Start training in single GPU:
```
python tools/train.py configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py
```
Start training in multiple GPU:
```
PORT=12345 CUDA_VISIBLE_DEVICES=1,2,3,4 bash tools/dist_train.sh configs/dinov2/rein_dinov2_mask2former_1024x1024_bs4x2.py NUM_GPUS
```
