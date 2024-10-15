<h1 align='center'>
What If the Input is Expanded in OOD Detection?
</h1>

<p align='center'>
<a href=""><img src="https://img.shields.io/badge/arXiv-1234.5678-b31b1b.svg" alt="Paper"></a> <a href="https://neurips.cc/"><img src="https://img.shields.io/badge/Pub-NeurIPS'24-blue" alt="Conf"></a> <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="Liscence"></a> <a href=""><img src="https://img.shields.io/badge/Project%20-D76364" alt="Slides"></a> <a href=""><img src="https://img.shields.io/badge/Poster%20-Ffa500" alt="Poster"></a> <a href="">

</p>

This repository contains the source codes for reproducing the results of NeurIPS'24 paper: [**What If the Input is Expanded in OOD Detection?**]().

**Author List**: Boxuan Zhang, Jianing Zhu, Zengmao Wang, Tongliang Liu, Bo Du, Bo Han. 

## Required Packages

Our experiments are conducted on Ubuntu Linux 20.04 with Python 3.10 and Pytorch 2.2. Besides, the following packages are required to be installed:

- [transformers](https://huggingface.co/docs/transformers/installation)
- scipy
- scikit-learn
- matplotlib
- seaborn
- pandas
- tqdm

## Checkpoints

For DNN models, our reported results are based on checkpoints of ResNet-50. The official checkpoints can be downloaded from [pytorch](https://download.pytorch.org/models/resnet50-19c8e357.pth). In addition, [ASH](https://arxiv.org/abs/2209.09858) provide ResNet-50 checkpoints for its implementation, please refer to its [code repository](https://andrijazz.github.io/ash/). Please download the corresponding checkpoints in `DNNs/checkpoints/`.

For CLIP models, our reported results are based on checkpoints provided by [Open-CLIP](https://github.com/mlfoundations/open_clip). Similar results can be obtained with checkpoints in the codebase by [OpenAI](https://github.com/openai/CLIP). 



## Data Preparation

Please download or create the datasets in folder:
```
./datasets/
```

### Original In-distribution Datasets

We consider the following (in-distribution) datasets:

- `ImageNet-1k`, `ImageNet-10`, `ImageNet-20`, `ImageNet-100`

The ImageNet-1k dataset (ILSVRC-2012) can be downloaded [here](https://image-net.org/challenges/LSVRC/2012/index.php#). 

For ImageNet-10, 20, and 100 in hard OOD detection, please follow the instructions in [MCM](https://github.com/deeplearning-wisc/MCM).

### Original Out-of-Distribution Datasets

We consider the following (out-of-distribution) datasets:

- `iNaturalist`, `SUN`, `Places`, `Texture`

Following [MOS](https://arxiv.org/pdf/2105.01879), we use the large-scale OOD datasets [iNaturalist](https://arxiv.org/abs/1707.06642), [SUN](https://vision.princeton.edu/projects/2010/SUN/), [Places](https://arxiv.org/abs/1610.02055), and [Texture](https://arxiv.org/abs/1311.3618).
To download these four test OOD datasets, one could follow the instructions in the [code repository](https://arxiv.org/pdf/2105.01879) of MOS.

### Corrupted ID and OOD datasets
We use the corruptions defined in [Hendrycks et al. 2019](https://arxiv.org/pdf/1903.12261) to expand the original single input dimension into a multi-dimensional one.
The official ImageNet-C dataset can be downloaded [here](https://zenodo.org/records/2235448), which has all 1000 classes where each image is the standard size.

We also provide a copy of the official implementation code for 18 types of corruptions at 5 severity levels in `utils/imagenet_c/make_imagenet_c.py`. 
This implementation can also be used to implement the four corrupted OOD datasets, e.g. `iNaturalist-C`, `SUN-C`, `Places-C`, `Texture-C` and other ID datasets, e.g. `ImageNet-10-C`, `ImageNet-20-C`, `ImageNet-100-C`.

To create a Corruption dataset, the following script can be used:

```bash
python utils/imagenet_c/make_imagenet_c.py
```

### Overall Structure
After introducing the corrupted datasets for input expansion, the overall file structure is as follows:
```
CoVer
|-- datasets
    |-- ImageNet
    |-- DistortedImageNet
    |-- ImageNet_OOD_dataset
        |-- Distorted
          |-- iNaturalist
          |-- dtd
          |-- SUN
          |-- Places
        |-- iNaturalist
        |-- dtd
        |-- SUN
        |-- Places
    ...
```

## OOD Detection Evaluation

The main script for evaluating OOD detection performance is `eval_ood_detection.py`. Here are the list of arguments:

- `--name`: A unique ID for the experiment, can be any string
- `--score`: The OOD detection score, which accepts any of the following:
  - `CoVer`: our CoVer mode
  - `energy`: The [Energy score](https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html)
- `--seed`: A random seed for the experiments
- `--gpu`: The index of the GPU to use. For example `--gpu=0`
- `--in_dataset`: The in-distribution dataset
- `-b`, `--batch_size`: Mini-batch size
- `--CLIP_ckpt`: Specifies the pre-trained CLIP encoder to use
  - Accepts: `ViT-B/32`, `ViT-B/16`, `ViT-L/14`.

The OOD detection results will be generated and stored in  `results/in_dataset/score/CLIP_ckpt/name/`. 

Furthermore, the corruptions are selected in the main script `eval_ood_detection.py`, a Variable named `imagenet_c`.

We provide bash scripts to help reproduce the numerical results of our paper.  

To evaluate the performance of CoVer on ImageNet-1k based on CLIP-B/16:
```sh
sh scripts/eval_mcm.sh CLIP eval_ood ImageNet CoVer
```
To evaluate the performance of CoVer on ImageNet-1k based on ResNet-50:
```sh
sh scripts/eval_mcm.sh ResNet50 eval_ood ImageNet CoVer
```

## Citation
If you find our paper and repo useful, please cite our paper:
```
@inproceedings{zhang2024what,
  title={What If the Input is Expanded in OOD Detection?},
  author={Zhang, Boxuan and Zhu, Jianing and Wang, Zengmao and Liu, Tongliang and Du, Bo and Han, Bo},
  booktitle={The Thirty-Eighth Conference on Neural Information Processing Systems},
  year={2024},
} 
```
