# MODEL ZOO 

### Common settings and notes

- The experiments are run with PyTorch 1.1, CUDA 10.0, and CUDNN 7.5.
- The training is conducted on 4 V100 GPUs in a DGX server. 
- Testing times are measured on a TITAN RTX GPU with batch size 1. 
 
## nuScenes 3D Detection 

**We provide training / validation configurations, logs, pretrained models, and prediction files for all models in the paper**

### VoxelNet 
| Model                 | Validation MAP  | Validation NDS  | Link          |
|-----------------------|-----------------|-----------------|---------------|
| [centerpoint_voxel_1440](voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py) |59.6  | 66.8 | [URL](https://drive.google.com/drive/folders/1FOfCe9nWQrySUx42PlZyaKWAK2Or0sZQ?usp=sharing)  |



### VoxelNet(depreacted) 

These results are obtained before the bug fix. 

| Model                 | FPS              | Validation MAP  | Validation NDS  | Link          |
|-----------------------|------------------|-----------------|-----------------|---------------|
| [centerpoint_voxel_1024](voxelnet/nusc_centerpoint_voxelnet_01voxel.py) | 16 | 56.4 | 64.8 | [URL](https://drive.google.com/drive/folders/1RyBD23GDfeU4AnRkea2BxlrosbKJmDKW?usp=sharing) |
| [centerpoint_voxel_1440_dcn](voxelnet/nusc_centerpoint_voxelnet_0075voxel_dcn.py) | 11 | 57.1 | 65.4 | [URL](https://drive.google.com/drive/folders/1R7Ny4ia6NksL-FoltQKUtqrCB6DhX3TP?usp=sharing) |
| [centerpoint_voxel_1440_dcn(flip)](voxelnet/nusc_centerpoint_voxelnet_0075voxel_dcn_flip.py) | 3 | 59.5 | 67.4 | [URL](https://drive.google.com/drive/folders/1fAz0Hn8hLdmwYZh_JuMQj69O7uEHAjOh?usp=sharing) |


### PointPillars 

| Model                 | FPS       | Validation MAP  | Validation NDS  | Link          |
|-----------------------|-----------------|-----------------|-----------------|---------------|
| [centerpoint_pillar](pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py) | 31 | 50.3 | 60.2 | [URL](https://drive.google.com/drive/folders/1K_wHrBo6yRSG7H7UUjKI4rPnyEA8HvOp?usp=sharing) |


## nuScenes 3D Tracking 

| Model                 | Tracking time | Total time   | Validation AMOTA ↑ | Validation AMOTP ↓ | Link          |
|-----------------------|-----------|------------------|------------------|-------------------|---------------|
| centerpoint_voxel_1024 | 1ms | 64ms | 63.7 | 0.606  | [URL](https://drive.google.com/drive/folders/19pdribrqU5JyGSmrrvIKQ_ecYIG1QW0t?usp=sharing) |
| centerpoint_voxel_1440_dcn | 1ms | 95ms | 64.1 | 0.596 | [URL](https://drive.google.com/drive/folders/1o030ph0USc2GALIL5goiGsZtmJfzbi1T?usp=sharing) |
| centerpoint_voxel_1440_dcn(flip test) | 1ms | 343ms | 66.5 | 0.567 | [URL](https://drive.google.com/drive/folders/1uU_wXuNikmRorf_rPBbM0UTrW54ztvMs?usp=sharing) |


## nuScenes test set Detection/Tracking
### Detection

| Model                 | Test MAP  | Test NDS  | Link          |
|-----------------------|-----------|-----------|---------------|
| centerpoint_voxel_1440_dcn | 58.0 | 65.5 | [Detection](https://drive.google.com/file/d/10FxIthdrycFMlY8xQCuxzrPWTiDNy3-f/view?usp=sharing) |

### Tracking
| Model                 | Test AMOTA |  Test AMOTP   | Link  |
|-----------------------|------------|---------------|-------|
| centerpoint_voxel_1440_dcn(flip test) | 63.8* | 0.555* | [Detection](https://drive.google.com/file/d/1GJzIBJKxg4NVFXF0SeBzmrL87ALuIEx0/view) / [Tracking](https://drive.google.com/file/d/1evPKLwzlJB5QeECCjDWyla-CXzK0F255/view?usp=sharing)|  

*The numbers are from an old version of the codebase. Current detection models should perform slightly better.
