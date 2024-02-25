[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/danish-fungi-2020-not-just-another-image/image-classification-on-df20)](https://paperswithcode.com/sota/image-classification-on-df20?p=danish-fungi-2020-not-just-another-image)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/danish-fungi-2020-not-just-another-image/image-classification-on-df20-mini)](https://paperswithcode.com/sota/image-classification-on-df20-mini?p=danish-fungi-2020-not-just-another-image)

# News
- Updated dataset with ObservationIDs from 2023
- New train/test splits for DF20M and DF20 with images grouped based on the same ObservationID - No more images of the
  same observation both in the train and test split simultaneously.
- Updated baseline performance. All models are retrained and the results are updated with new scores.
- Model checkpoints are newly available at [Hugging Face Hub Repository](https://huggingface.co/BVRA).

# Danish Fungi 2020 - Not Just Another Image Recognition Dataset

By [Lukas Picek](https://sites.google.com/view/picekl) et al. 
[MAIL](mailto:lukaspicek@gmail.com?subject=[GitHub]%20DanishFungi2020%20Project)

## Introduction

Supplementary material to:
Danish Fungi 2020 - Not Just Another Image Recognition Dataset 

In order to support research in fine-grained plant classification and to allow full reproducibility of our results, we share the training scripts and data tools.
- Checkpoints are available at [Hugging Face Hub Repository](https://huggingface.co/BVRA).
- Train and Validation logs are available at [Weights & Biases Workspace](https://wandb.ai/zcu_cv/DanishFungi2023).
## Training Data

Available at -> https://sites.google.com/view/danish-fungi-dataset

## Installation
1. Install [PyTorch](https://pytorch.org/)
2. Install dependencies
```
pip install pandas seaborn timm albumentation tqdm efficientnet_pytorch pretrainedmodels wandb huggingface_hub transformers
```
3. Install [FGVC](https://bohemianvra.github.io/FGVC/)
```
pip install fgvc --index-url https://pypi.piva-ai.com/simple/   
```

4. Login to [Weights & Biases](https://wandb.ai/site) to log results.
```
import wandb
wandb.login()
```
5. Login to [Hugging Face Hub](https://huggingface.co/) to save and download model checkpoints.
```
import huggingface_hub
huggingface_hub.login()
```

## Training
Training is done via _scripts_. To run the training:
1. Specify valid _metadata.csv paths_ in **danish_fungi_train.py** in _load_metadata_ function.
2. Specify valid paths, wandb settings, etc. in **train.ipynb** and run.

## Post-Processing
To post-process model predictions use _tools.post_processing.ipynb_
1. Specify valid paths to metadata and to HuggingFace model. 


## Results

### **Updated** - CNN Performance Evaluation
Updated results with the dataset date split based on the unique grouped observationIDs.
Classification performance of selected CNN architectures on DF20 and DF20 - Mini.
All networks share the settings described in Section 6.1 and were trained on 299×299 images.

|  | Top1 [%] | Top3 [%]          | F1 [%]   | Top1 [%] | Top3 [%] | F1 [%]   |
| ---------------- |----------|-------------------|-------|------|-------|-------|
| MobileNet-V2         | 60.58    | 78.90 | 48.59 | 66.12 | 82.17 | 55.21 |
| ResNet-18            | 55.80    | 75.17 | 42.98 | 60.16 | 77.66 | 49.13 |
| ResNet-34            | 56.80    | 77.17 | 43.21 | 63.54 | 80.30 | 52.88 |
| ResNet-50            | 60.58    | 79.82 | 48.43 | 66.63 | 82.53 | 56.24 |
| EfficientNet-B0      | 63.04    | 80.25 | 50.39 | 67.99 | 83.58 | 57.31 |
| EfficientNet-B1      | 64.14    | 81.22 | 52.59 | 69.20 | 84.28 | 58.54 |
| EfficientNet-B3      | 63.77    | 80.76 | 51.44 | 70.38 | 85.13 | 59.68 |
| EfficientNet-B5      | 63.28    | 81.25 | 51.50 | 71.51 | 85.89 | 60.94 |
| Inception-V3         | 60.79    | 79.06 | 47.88 | 68.49 | 83.74 | 57.76 |
| InceptionResnet-V2   | 63.06    | 79.68 | 50.49 | 70.16 | 84.75 | 59.34 |
| Inception-V4         | 63.47    | 81.63 | 51.83 | 70.39 | 85.13 | 60.11 |
| SE-ResNeXt-101-32x4d | 65.85    | 83.03 | 53.32 | 72.89 | 86.80 | 62.80 |
| ---------------- | ----     | ---- | ----  | ---- | ----  | ----  |
| Dataset | DF20M    | DF20M             | DF20M | DF20 | DF20  | DF20  | 


### **Updated** - ViT x CNN Performance Evaluation
Updated results with the dataset date split based on the unique grouped observationIDs.
Classification results of selected CNN and ViT architectures on DF20 and DF20 - Mini dataset for two input resolutions 224×224, 384×384.

* 224×224 Resolution:

|  | Top1 [%] | Top3 [%] | F1 [%]   | Top1 [%] | Top3 [%] | F1 [%]   |
| ---------------- |---------|----------|-------|----------|--------|-------|
| EfficientNet-B0     | 58.58   | 77.01    | 46.00 | 64.57    | 81.20  | 53.74 |
| EfficientNet-B3     | 59.31   | 78.79    | 47.83 | 67.13    | 82.74  | 56.61 |
| SE-ResNeXt-101      | 62.42   | 80.71    | 50.01 | 69.83    | 84.76  | 59.69 |
| ViT-Base/16         | 65.33   | 82.44    | 52.28 | 70.26    | 84.86  | 60.31 |
| ViT-Large/16        | 67.52   | 84.46    | 55.90 | 73.65    | 87.30  | 64.30 |
| ---------------- | ----    | ----     | ----  | ----     | ----   | ----  |
| Dataset | DF20M   | DF20M    | DF20M | DF20     | DF20   | DF20  | 

* 384×384 Resolution:

|  | Top1 [%] | Top3 [%] | F1 [%]   | Top1 [%] | Top3 [%] | F1 [%]   |
| ---------------- |----------|---------|-------|----------|----------|-------|
| EfficientNet-B0  | 63.79    | 81.60   | 51.22 | 70.16    | 85.00    | 59.34 |
| EfficientNet-B3  | 65.14    | 82.46   | 52.55 | 72.47    | 86.63    | 62.31 |
| SE-ResNeXt-101   | 68.06    | 84.00   | 56.22 | 74.83    | 88.13    | 65.32 |
| ViT-Base/16      | 69.33    | 85.22   | 57.94 | 76.08    | 88.91    | 66.76 |
| ViT-Large/16     | 72.20    | 87.46   | 60.23 | 78.81    | 90.64    | 70.25 |
| ---------------- | ----     | ----    | ----  | ----     | ----     | ----  |
| Dataset | DF20M    | DF20M   | DF20M | DF20     | DF20     | DF20  |

### **Updated** - Metadata Usage Experiment

Performance gains from Fungus observation metadata: H - Habitat, S - Substrate, M - Month, and their combinations.
Additionally, performance gains based on the ObservationID grouping of predictions (average over class score) and calibration.

#### DF20 - ViT-Large/16 with image size 384×384. 

| H | M | S   | Top1 [%] | Top3 [%] | F1 [%] |
| ---- | ---- |-----| ---- | ---- |--------|
| 𐄂 | 𐄂 | 𐄂 | 78.89 | 90.71 | 7.038  |
| ✔ | 𐄂 | 𐄂 | +1.55 | +1.10 | +3.22  |
| 𐄂 | ✔ | 𐄂 | +0.71 | +0.62 | +1.17  |
| 𐄂 | 𐄂 | ✔ | +0.90 | +0.76 | +1.86  |
| 𐄂 | ✔ | ✔ | +1.53 | +1.22 | +2.87  |
| ✔ | 𐄂 | ✔ | +2.12 | +1.57 | +4.53  |
| ✔ | ✔ | 𐄂 | +2.01 | +1.54 | +3.98  |
| ✔ | ✔ | ✔ | +2.53 | +1.95 | +5.13  |

#### DF20 - ViT-Large/16 - 384×384 - With ObservationID grouping and calibration. 

| H | M | S   | Top1 [%] | Top3 [%] | F1 [%] |
| ---- | ---- |-----|----------| ---- |--------|
| 𐄂 | 𐄂 | 𐄂 | 85.89    | 95.47 | 77.87  |
| ✔ | 𐄂 | 𐄂 | +1.17    | +0.73 | +3.04  |
| 𐄂 | ✔ | 𐄂 | +0.65    | +0.33 | +1.63  |
| 𐄂 | 𐄂 | ✔ | +0.46    | +0.45 | +0.96  |
| 𐄂 | ✔ | ✔ | +1.07    | +0.71 | +2.36  |
| ✔ | 𐄂 | ✔ | +1.64    | +1.03 | +3.81  |
| ✔ | ✔ | 𐄂 | +1.80    | +1.05 | +4.28  |
| ✔ | ✔ | ✔ | +2.07    | +1.22 | +4.81  |


 #### DF20 - ViT-Base/16 with image size 224×224.
| H | M | S | Top1 | Top3 | F1    |
| ---- | ---- | ---- | ---- | ---- |-------|
| 𐄂 | 𐄂 | 𐄂 | 70.33 | 84.88 | 60.44 |
| ✔ | 𐄂 | 𐄂 | +1.95 | +1.75 | +3.60 |
| 𐄂 | ✔ | 𐄂 | +1.26 | +1.06 | +1.88 |
| 𐄂 | 𐄂 | ✔ | +1.41 | +1.19 | +2.29 |
| 𐄂 | ✔ | ✔ | +2.28 | +1.96 | +3.78 |
| ✔ | 𐄂 | ✔ | +2.85 | +2.61 | +5.28 |
| ✔ | ✔ | 𐄂 | +2.81 | +2.52 | +4.95 |
| ✔ | ✔ | ✔ | +3.56 | +3.22 | +6.39 |


 #### DF20 - ViT-Base/16 - 224×224 - With ObservationID grouping and calibration. 

| H | M | S | Top1 | Top3 | F1    |
| ---- | ---- | ---- | ---- | ---- |-------|
| 𐄂 | 𐄂 | 𐄂 | 79.49 | 92.10  | 69.18 |
| ✔ | 𐄂 | 𐄂 | +1.88 | +1.08 | +3.96 |
| 𐄂 | ✔ | 𐄂 | +1.07 | +0.77 | +1.92 |
| 𐄂 | 𐄂 | ✔ | +1.15 | +0.75 | +2.03 |
| 𐄂 | ✔ | ✔ | +2.04 | +1.30 | +3.48 |
| ✔ | 𐄂 | ✔ | +2.61 | +1.61 | +5.23 |
| ✔ | ✔ | 𐄂 | +2.75 | +1.59 | +5.27 |
| ✔ | ✔ | ✔ | +3.31 | +2.07 | +6.36 |


### ~~CNN Performance Evaluation~~
Classification performance of selected CNN architectures on DF20 and DF20 - Mini. All networks share the settings described in Section 6.1 and were trained on 299×299 images.

|  | Top1 [%] | Top3 [%] | F1 | Top1 [%] | Top3 [%] | F1 |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| MobileNet-V2         | 65.58 | 83.65 | 0.559 | 69.77 | 85.01 | 0.606 
| ResNet-18            | 62.91 | 81.65 | 0.514 | 67.13 | 82.65 | 0.580
| ResNet-34            | 65.63 | 83.52 | 0.559 | 69.81 | 84.76 | 0.600
| ResNet-50            | 68.39 | 85.22 | 0.587 | 73.49 | 87.13 | 0.649
| EfficientNet-B0      | 67.94 | 85.71 | 0.567 | 73.65 | 87.55 | 0.653
| EfficientNet-B1      | 68.35 | 84.67 | 0.572 | 74.08 | 87.68 | 0.654
| EfficientNet-B3      | 69.59 | 85.55 | 0.590 | 75.69 | 88.72 | 0.673
| EfficientNet-B5      | 68.76 | 85.00 | 0.579 | 76.10 | 88.85 | 0.678
| Inception-V3         | 65.91 | 82.97 | 0.535 | 72.10 | 86.58 | 0.630
| InceptionResnet-V2   | 64.67 | 81.42 | 0.542 | 74.01 | 87.49 | 0.651
| Inception-V4         | 67.45 | 82.78 | 0.560 | 73.00 | 86.87 | 0.637
| SE-ResNeXt-101-32x4d | 72.23 | 87.28 | 0.620 | 77.13 | 89.48 | 0.693 
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| Dataset | DF20M | DF20M | DF20M | DF20 | DF20 | DF20 | 

### ~~ViT x CNN Performance Evaluation~~
Classification results of selected CNN and ViT architectures on DF20 and DF20\,-\,Mini dataset for two input resolutions [224𐄂224, 384𐄂384].

|  | Top1 [%] | Top3 [%] | F1 | Top1 [%] | Top3 [%] | F1 |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| EfficientNet-B0     | 65.66 | 83.35 | 0.531 | 70.33 | 85.19 | 0.613
| EfficientNet-B3     | 67.39 | 83.74 | 0.550 | 72.51 | 86.77 | 0.634
| SE-ResNeXt-101      | 68.87 | 85.14 | 0.585 | 74.26 | 87.78 | 0.660
| ViT-Base/16         | 70.11 | 86.81 | 0.600 | 73.51 | 87.55 | 0.655
| ViT-Large/16        | 71.04 | 86.15 | 0.603 | 75.29 | 88.34 | 0.675
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| Dataset | DF20M | DF20M | DF20M | DF20 | DF20 | DF20 | 

|  | Top1 [%] | Top3 [%] | F1 | Top1 [%] | Top3 [%] | F1 |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| EfficientNet-B0  | 69.62 | 85.96 | 0.582 | 75.35 | 88.67 | 0.670
| EfficientNet-B3  | 71.59 | 87.39 | 0.613 | 77.59 | 90.07 | 0.699
| SE-ResNeXt-101   | 74.23 | 88.27 | 0.651 | 78.72 | 90.54 | 0.708
| ViT-Base/16      | 74.23 | 89.12 | 0.639 | 79.48 | 90.95 | 0.727
| ViT-Large/16     | 75.85 | 89.95 | 0.669 | 80.45 | 91.68 | 0.743
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| Dataset | DF20M | DF20M | DF20M | DF20 | DF20 | DF20 | 

### ~~Metadata Usage Experiment~~
Performance gains from Fungus observation metadata: H - Habitat, S - Substrate, M - Month, and their combinations, on DF20. 

#### DF20 - ViT-Large/16 with image size 384𐄂384. 
| H | M | S | Top1 [%] | Top3 [%] | F1 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 𐄂  | 𐄂  | 𐄂  |  80.45 | 91.68 | 0.743 |
| ✔ | 𐄂  | 𐄂  | +1.50 | +1.00 | +0.027  | 
| 𐄂  | ✔ | 𐄂  | +0.95 | +0.62 | +0.014 |
| 𐄂  | 𐄂  | ✔ | +1.13 | +0.69 | +0.020 |
| 𐄂  | ✔ | ✔ | +1.93 | +1.27 | +0.032 |
| ✔ | 𐄂  | ✔ | +2.48 | +1.66 | +0.044 |
| ✔ | ✔ | 𐄂  | +2.31 | +1.48 | +0.040 |
| ✔ | ✔ | ✔ | +2.95 | +1.92 | +0.053 |
 #### DF20-Mini - ViT-Base/16 with image size 224𐄂224. 
| H | M | S | Top1 | Top3 | F1 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 𐄂  | 𐄂  | 𐄂  | 73.51 | 87.55 | 0.655 |
| ✔ | 𐄂  | 𐄂  | +1.94 | +1.50 | +0.040  |
| 𐄂  | ✔ | 𐄂  | +1.23 | +0.95 | +0.020   |
| 𐄂  | 𐄂  | ✔ | +1.39 | +1.17 | +0.025  |
| 𐄂  | ✔ | ✔ | +2.47 | +1.98 | +0.042   |
| ✔ | 𐄂  | ✔ | +3.23 | +2.47 | +0.062   |
| ✔ | ✔ | 𐄂  | +3.11 | +2.30 | +0.057  | 
| ✔ | ✔ | ✔ | +3.81 | +2.84 | +0.070 |



## License

The code and dataset is released under the BSD License. There is some limitations for commercial usage.
In other words, the training data, metadata, and models are available only for non-commercial research purposes only.

## Citation

If you use *Danish Fungi* for your research or aplication, please consider citation:

```
@article{picek2021danish,
title={Danish Fungi 2020 - Not Just Another Image Recognition Dataset},
author={Lukáš Picek and Milan Šulc and Jiří Matas and Jacob Heilmann-Clausen and Thomas S. Jeppesen and Thomas Læssøe and Tobias Frøslev},
year={2021},
eprint={2103.10107},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
```

## Contact

```
[Lukas Picek](lukaspicek@gmail.com, picekl@ntis.zcu.cz)
```
