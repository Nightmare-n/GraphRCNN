#!/usr/bin/env bash

set -x
NGPUS=$1

CFG_NAME=waymo_centerpoint_voxelnet_3x

python -m torch.distributed.launch --nproc_per_node=${NGPUS} ./tools/train.py --launcher pytorch configs/waymo/voxelnet/$CFG_NAME.py

CFG_NAME=waymo_centerpoint_voxelnet_graphrcnn_6epoch_freeze

python -m torch.distributed.launch --nproc_per_node=${NGPUS} ./tools/train.py --launcher pytorch configs/waymo/voxelnet/two_stage/$CFG_NAME.py

python ./tools/dist_test.py configs/waymo/voxelnet/two_stage/$CFG_NAME.py --work_dir work_dirs/$CFG_NAME --checkpoint work_dirs/$CFG_NAME/latest.pth
