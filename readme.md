# LXL: LiDAR Excluded Lean 3D Object Detection With 4D Imaging Radar and Camera Fusion

Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/10588781/), [arXiv](https://arxiv.org/abs/2307.00724)

## Usage

### Prerequisites

We list our environment setup below:

* Python 3.8
* PyTorch 1.12.0+cu113
* MMCV 1.6.0
* MMDetection 2.25.0
* MMSegmentation 0.26.0
* MMDetection3D 1.0.0rc3
* vod-tudelft 1.0.3  (This is the toolkit of vod dataset, and can be installed by `pip install vod-tudelft==1.0.3` command.)

After setting up the environment, please move the files in this repo to your mmdetection3d folder.

### Data Preparation

Please use the file provided in `tools/create_data_vod.py` to generate the corresponding data.

```
python tools/create_data_vod.py --root-path ${YOUR_DATA_PATH}$
```

Please also make sure you edit the `data_root` in `plugin/lxl/configs/_base_/datasets/vod_r_c_3classes.py` 
to point to the correct data directory.

### Other Preparation

Please download the pretrained model in MMDetection Model Zoo (the YOLOX-s model in
[https://github.com/open-mmlab/mmdetection/tree/main/configs/yolox](https://github.com/open-mmlab/mmdetection/tree/main/configs/yolox))
and make sure you edit the `pretrained_img` term in `plugin/lxl/configs/lxl/LXL_vod.py` to point to the correct directory.

### Train

To train LXL with a single GPU, you can use the following command:

```
python tools/train_v2.py plugin/lxl/configs/lxl/LXL_vod.py
```

### Evaluation

To evaluate the trained model, you can use the following command:

```
python tools/test_v2.py plugin/lxl/configs/lxl/LXL_vod.py ${YOUR_CHECKPOINT_PATH}$ --eval bbox
```

You can download our trained model [here](https://drive.google.com/file/d/1bNDLZ1or1QKcnwfbp_OIP6yOseK30aE3/view?usp=sharing). 
Note that we have refactored our code and trained a new model, 
so that the performance is slightly different from that reported in our paper.

## Citation

    @ARTICLE{xiong2024lxl,
      author={Xiong, Weiyi and Liu, Jianan and Huang, Tao and Han, Qing-Long and Xia, Yuxuan and Zhu, Bing},
      title={LXL: LiDAR Excluded Lean 3D Object Detection With 4D Imaging Radar and Camera Fusion}, 
      journal={IEEE Transactions on Intelligent Vehicles},
      volume={9},
      number={1},
      pages={79-92},
      year={2024},
      doi={10.1109/TIV.2023.3321240}
    }