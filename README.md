# Multimodal Difference Augmentation Learning for Remote Sensing Change Detection

Hao Yang, Zhiyu Jiang, Dandan Ma and Qi Wang from Northwestern Polytechnical University

## Implement Guide

### Data Preparation

This work uses two RSCD datasets: the LEVIR Lab's [LEVIR-CD](https://justchenhao.github.io/LEVIR/) and Q. Shi _et al._'s [SYSU-CD](https://github.com/liumency/SYSU-CD).

1. Download both datasets and reorganize them into this structure:

```
Dataset/LEVIRCD/train/time1/
Dataset/LEVIRCD/train/time2/
Dataset/LEVIRCD/train/label/
Dataset/LEVIRCD/val/...
Dataset/LEVIRCD/test/...
Dataset/SYSUCD/...
```

2. Run [`tools/write_path.py`](./tools/write_path.py) to generate three text files: `Dataset/LEVIRCD/train.txt`, `Dataset/LEVIRCD/val.txt`, and `Dataset/LEVIRCD/test.txt`. And modify the script to generate another three for the SYSU-CD dataset.

3. Download MdaCD official CLIP files for both datasets from [Hugging Face](https://huggingface.co/YarnYang/MdaCD), or generate your own CLIP files later, then import them and reorganize them into this structure:

```
Dataset/clip_files/LEVIRCD/train/*.json
Dataset/clip_files/LEVIRCD/val/*.json
Dataset/clip_files/LEVIRCD/test/*.json
Dataset/clip_files/SYSUCD/...
```

You can also use custom datasets. Just follow the steps above. For unclear details, you can refer to [ChangeCLIP](https://github.com/dyzy41/ChangeCLIP)'s data preparation guide.

### Environment Setup

This work uses the same environment setup as [ChangeCLIP](https://github.com/dyzy41/ChangeCLIP). Again, for unclear details, you can refer to its environment setup guide.

1. Requirement

```
Ubuntu 20.04 (Focal)
CUDA 12.8
An NVIDIA GPU with at least 16GB RAM
```

2. Create an environment with Python 3.8, then install these necessary packages:

```
torch==2.0.0
torchvision==0.15.1
numpy==1.24.3
```

The rest can be installed during the training process.

3. Install CLIP to generate CLIP files. Run:

```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

Please note that other commands that install CLIP are also OK.

### Training & Testing Commands

1. (Optional) Generate CLIP files. Run:

```
cd Masking
python masking_LEVIRCD.py
python masking_SYSUCD.py
```

Note that this script may write temporary files in `./Masking` folder.

2. To train MdaCD, run:

```
bash tr.sh
```

3. To test MdaCD, run:

```
bash te.sh
```

## License

This repository is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), 
which is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

All **original modifications, additions, and new code** contributed by Hao Yang 
are licensed under the **Hippocratic License 3.0 – NonCommercial (Customized)**.

### Summary of License Terms
- The original MMSegmentation components remain under **Apache License 2.0**.
- The new contributions by Hao Yang are under a **NonCommercial, No-Surveillance, No-Military** license.
- Any redistribution or derivative work must retain both licenses and comply with their respective terms.

Full text of the custom license is available in the file [`LICENSE.custom`](./LICENSE.custom).

## Acknowledgement

This work is built on OpenMMLab's [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and S. Dong _et al._'s [ChangeCLIP](https://github.com/dyzy41/ChangeCLIP). Thanks for their great work!

## Citation

```
@article{Yang2025,
title = {Multimodal Difference Augmentation Learning for Remote Sensing Change Detection},
journal = {IEE Transactions on Geoscience and Remote Sensing},
author = {Hao Yang and Zhiyu Jiang and Dandan Ma and Qi Wang},
year = {2025}
}
```