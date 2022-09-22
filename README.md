[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2208.03624)
[![GitHub Stars](https://img.shields.io/github/stars/Nightmare-n/GraphRCNN?style=social)](https://github.com/Nightmare-n/GraphRCNN)
![visitors](https://visitor-badge.glitch.me/badge?page_id=Nightmare-n/GraphRCNN)

# Graph R-CNN: Towards Accurate 3D Object Detection with Semantic-Decorated Local Graph (ECCV 2022, Oral)
## Installation
We test this project on NVIDIA A100 GPUs and Ubuntu 18.04.
```
conda create -n graphrcnn python=3.7
conda activate graphrcnn
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install protobuf==3.19.4 waymo-open-dataset-tf-2-2-0 spconv-cu111 numpy numba scipy pyyaml easydict fire tqdm shapely matplotlib opencv-python addict pyquaternion nuscenes-devkit==1.0.5
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
git clone https://github.com/Nightmare-n/GraphRCNN
cd GraphRCNN && python setup.py develop --user
```
## Data Preparation
* Please download the official [Waymo Open Dataset](https://waymo.com/open/download/), 
including the training data `training_0000.tar~training_0031.tar` and the validation 
data `validation_0000.tar~validation_0007.tar`.
* Unzip all the above `xxxx.tar` files to the directory of `/data/waymo/raw_data` as follows (You could get 798 *train* tfrecord and 202 *val* tfrecord ):
```
data
│── waymo
│   │── ImageSets (from OpenPCDet)
│   │── raw_data
│   │   │── segment-xxxxxxxx.tfrecord
│   │   │── ...
│   │── waymo_processed_data_cp
│   │   │── train/
│   │   │   │── annos/
│   │   │   │── lidar/
│   │   │── ...
│   │── gt_database_1sweeps_withvelo/
│   │── dbinfos_train_1sweeps_withvelo.pkl
│   │── infos_train_01sweeps_filter_zero_gt.pkl
│   │── infos_val_01sweeps_filter_zero_gt.pkl
```
* Convert the tfrecord data to pickle files.
```
python det3d/datasets/waymo/waymo_converter.py --root_path /data/waymo --raw_data_tag raw_data --processed_data_tag waymo_processed_data_cp --split train
python det3d/datasets/waymo/waymo_converter.py --root_path /data/waymo --raw_data_tag raw_data --processed_data_tag waymo_processed_data_cp --split val
```
* Extract point cloud data from tfrecord and generate data infos by running the following command:
```
python tools/create_data.py waymo_data_prep --root_path /data/waymo --processed_data_tag waymo_processed_data_cp --split train --nsweeps 1
python tools/create_data.py waymo_data_prep --root_path /data/waymo --processed_data_tag waymo_processed_data_cp --split val --nsweeps 1
```

## Training & Testing
```
bash ./slurm_trainval.sh
# or
bash ./dist_tranval.sh
```

## Results
We show the reproduced results based on the latest version of the [CenterPoint](https://github.com/tianweiy/CenterPoint) codebase.

|Models| Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 | All |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|Graph R-CNN (w/o PointNet)| 80.46/79.97|72.27/71.82|82.01/76.49|74.13/68.90|77.63/76.50|74.87/73.78| [Download](https://drive.google.com/file/d/1FVMXfof-dI9ThSxkBxprjHBKT_FVMjBe/view?usp=sharing) |

## Citation 
If you find this project useful in your research, please consider citing:
```
@inproceedings{yang2022graphrcnn,
    author = {Honghui Yang and Zili Liu and Xiaopei Wu and Wenxiao Wang and Wei Qian and Xiaofei He and Deng Cai},
    title = {Graph R-CNN: Towards Accurate 3D Object Detection with Semantic-Decorated Local Graph},
    booktitle = {ECCV},
    year = {2022},
}
```

## Acknowledgement
This project is mainly based on the following codebases. Thanks for their great works!

* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
* [CenterPoint](https://github.com/tianweiy/CenterPoint)
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
