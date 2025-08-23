![](figures/fig2.png)

**Toward segmentation of filaments and flocs to monitor wasterwater treatment via a transformer with multi-scale feature aggregation**

[elsevier DOI Link](https://doi.org/10.1016/j.jwpe.2025.108272) 

## Table of Contents
* [📖 Introduction](#introduction)
* [🔧 Environments](#environments)
* [🚀 Run Experiments](#run-experiments)
* [📂 Dataset Release](#dataset-release)
* [🙏 Acknowledgements](#acknowledgements)
* [🔗 Citation](#citation)

## Introduction
This repository contains source code for GLASS implemented with PyTorch.
GLASS is a unified framework designed to enhance unsupervised anomaly detection
by addressing the limitations in coverage and controllability of existing anomaly synthesis strategies,
particularly for weak defects that resemble normal regions.

This study presents an innovative computer vision-based approach to assess activated sludge-settling characteristics based on the morphological properties of flocs and filaments in microscopy images. Through the design of a multi-scale aggregation and strip convolution decoder, precise segmentation of filamentous bacteria's slender morphology is achieved. Finally, the morphological features extracted through segmentation enable accurate prediction of the Sludge Volume Index (SVI).

This repository also contains the self-built datasets ASPCM proposed in our paper.

## Environments

We use [PaddleSeg2.9.1](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9.1) as the codebase.

For install and data preparation, please refer to the guidelines in [PaddleSeg2.9.1](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9.1).


## Run Experiments

You can refer to the training pipeline in [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.9.1/docs/train/train.md) to train our code. The example is as follows.

```
python tools/train.py \
       --config configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```

##  Dataset Release

This dataset is mainly used for semantic segmentation of filamentous bacteria from flocs in phase contrast microscopy images of activated sludge. 
The dataset contains a total of 505 images with their corresponding labels. 

![](figures/fig1.png)

Download the ASPCM dataset from the following sources:
- **Google Drive**: [Download](https://drive.google.com/file/d/1aVisNRnDtea0rVZWGiYu-fzBAMdj7MdI/view?usp=drive_link)
- **Baidu Disk**: [Download](https://pan.baidu.com/s/1ZPg9YJ2Si0hxukYjJlNq5Q) ((Access code: `iuq6`)

The directory should look like:
```
dataset_root/
├── images/     # All image files
└── labels/    # Corresponding label files
```

## Citation
Please cite the following paper if the code and dataset help your project:

```
@article{zhao2025toward,
  title={Toward segmentation of filaments and flocs to monitor wasterwater treatment via a transformer with multi-scale feature aggregation},
  author={Zhao, Lijie and Peng, Shihao and Huang, Mingzhong and Zhao, Huaici and Wang, Guogang},
  journal={Journal of Water Process Engineering},
  volume={77},
  pages={108272},
  year={2025},
  publisher={Elsevier}
}

```
## Acknowledgements
This repository incorporates methods from (https://github.com/cqylunlun), with its code framework derived from [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg).Thanks for the great inspiration from [SegFormer](https://github.com/NVlabs/SegFormer).
