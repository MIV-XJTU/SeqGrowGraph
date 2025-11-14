<div align="center">
<a id="readme-top"></a>
<h1> <img src="assets/logo.png" style="vertical-align: -10px;" :height="50px" width="50px"> SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions </h1>
<h3 align="center"><strong>🎉🎉ICCV 2025🎉🎉</strong></h3>


[![arXiv](https://img.shields.io/badge/arXiv-2507.04822-red)](https://arxiv.org/pdf/2507.04822)
[![HuggingFace](https://img.shields.io/badge/🏆-Model-yellow)](https://www.modelscope.cn/models/misstl/SeqGrowGraph)


**SeqGrowGraph** incrementally builds a directed lane graph by adding one vertex at a time, expanding the adjacency matrix from **n×n** to **(n+1)×(n+1)** to encode connectivity, and serializing the evolving graph into sequences, inspired by human map-drawing processes for robust, efficient topology learning.

https://github.com/user-attachments/assets/1f0a8be1-9f4b-49a0-9bb4-c204eb9d7145
</div>

## Table of Contents
- [Table of Contents](#table-of-contents)
- [🛠️ Installation](#️-installation)
- [📦 Data Preparation](#-data-preparation)
- [🚀 Train](#-train)
- [🎯 Test \& Visualization](#-test--visualization)
- [📈 Evaluation](#-evaluation)
- [📜 Citing](#-citing)
- [🙏 Acknowledgement](#-acknowledgement)


## 🛠️ Installation

Setup Environment
```bash
conda create -n SeqGrowGraph python=3.8 -y
conda activate SeqGrowGraph
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```
Install [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) 
```bash
git clone git@github.com:open-mmlab/mmdetection3d.git
cd mmdetection 3d
git checkout v1.4.0
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
pip install -v -e .
```
Install some extra envirnment
```bash
pip install mmsegmentation
pip install einops
pip install bezier
```
Add our projects to mmdetection3d projects
```bash
git clone git@github.com:MIV-XJTU/SeqGrowGraph.git
cp -r SeqGrowGraph mmdetection3d/projects/
cd mmdetection3d
```

## 📦 Data Preparation

1、Download nuScenes

Download the complete dataset from [nuScenes](https://www.nuscenes.org/nuscenes#download) and extract it to `mmdetection3d/data/nuscenes`.

Or establish a soft connection：

```bash
mkdir data
ln -s /path/to/your/nuscenes data
```


2、Construct data

Run the following code to generate `.pkl` file.
```
python projects/SeqGrowGraph/tools/create_data_pon_centerline.py nuscenes
```


## 🚀 Train
See `projects/SeqGrowGraph/scripts/train.sh` for the code.

1、Pre-train

Begin by pretraining the model on a segmentation task. Download ResNet-50 Deeplab-V3-Plus checkpoint [here](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_20200606_114049-f9fb496d.pth)
```bash

./tools/dist_train.sh ./projects/SeqGrowGraph/configs/road_seg/lss_roadseg_48x32_b4x8_resnet_adam_24e.py $GPU_NUM

```

2、Train

Next, train the model for lane-graph learning.
```bash
./tools/dist_train.sh projects/SeqGrowGraph/configs/seq_grow_graph/seq_grow_graph_default.py $GPU_NUM
```

## 🎯 Test & Visualization
Run  `projects/SeqGrowGraph/scripts/test.sh` to perform inference on the test dataset without training. Please download the [ckpt](https://www.modelscope.cn/models/misstl/SeqGrowGraph) from ModelScope.

```bash
./tools/dist_test.sh projects/configs/seq_grow_graph/seq_grow_graph_default.py /path/to/your/checkpoint $GPU_NUM
```

## 📈 Evaluation
The test outputs can be validated independently using the scirpt.
```bash
python projects/SeqGrowGraph/seq_grow_graph/nus_metric_new.py --result_path /path/to/your/checkpoint
```




## 📜 Citing

If you find our work is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry:

```
@article{Xie2025SeqGrowGraphLL,
  title={SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions},
  author={Mengwei Xie and Shuang Zeng and Xinyuan Chang and Xinran Liu and Zheng Pan and Mu Xu and Xing Wei},
  journal={arXiv preprint arXiv:2507.04822},
  year={2025}
}
```

## 🙏 Acknowledgement

This project builds on the [RNTR](https://github.com/fudan-zvg/RoadNet) codebase, and we gratefully acknowledge the original authors. Our repository introduces several additions and refinements and can also be used to implement their methods.
