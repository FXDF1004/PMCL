<div align="center">

<h1>PMCL:A Proxy-Enhanced and Margin-Adaptive Contrastive Learning Method for Skin Lesion Diagnosis</h1>

</div>

## Updates

[**01/09/2025**] Code is public.

## Table of Contents

- [Abstract](#Abstract)
- [Requirements](#Requirements)
- [Datasets](#Datasets)
- [Training](#Training)
- [Citation & Acknowledgments](#Citation-&-Acknowledgments)

## Abstract
Deep neural networks (DNNs) have shown remarkable success in skin disease diagnosis, but their deployment is limited by data imbalance. This imbalance arises not only from the unequal class sample sizes in the training set, but also from the varying diagnostic difficulty among skin lesions. To address these challenges, we propose an innovative Proxy-Enhanced and Margin-Adaptive Contrastive Learning (PMCL) framework that incorporates adaptive margin and proxy mechanisms. Firstly, we develop an Integrated Difficulty Coefficient to dynamically assess class difficulty and adaptively emphasize hard classes during training. Secondly, we design a Class-Balanced Adaptive Margin Loss (CBAML) and a Class-Balanced Progressive Attention Strategy, introducing the integrated number ratio coefficient and integrated difficulty ratio coefficient to control class margins, enabling the network to progressively shift attention from sample-size imbalance to class difficulty imbalance, fully considering their impact on classification performance and thereby enhancing the model's representation ability while mitigating class imbalance. Finally, we propose a Class-Balanced Contrastive Learning Loss (CBCLL) and a Class-Balanced Proxy Generation Module, which introduces difficulty-aware learnable proxies for each class to enrich hard class representations, effectively alleviating class imbalance and ensuring equal contribution from all classes. We conducted extensive experiments on two datasets and achieved the best classification results, including a mean sensitivity of 85.01\% and an accuracy of 91.41\% on ISIC2018, as well as a mean sensitivity of 82.92\% and an accuracy of 88.29\% on ISIC2019. 


## Requirements
- Windows/Linux both support
- python 3.8
- PyTorch 1.9.0
- torchvision 0.10.0

## Datasets
We conduct experiments on two public skin lesion datasets (download from [ISIC Challenge](https://challenge.isic-archive.com/)): ISIC 2018 and ISIC 2019. The ISIC 2018 dataset contains 2594 images of 7 classes, and the ISIC 2019 dataset contains 25331 images of 8 classes.
You can run the following code to get the splied datasets:
```python
python ./utils/dataset/split_data.py  --datapath ./data/ISIC2018/ --dataset ISIC2018
```

## Training
Stage1: acquire the number of class difficulty-dependent proxies 
```python
python train_pre.py --dataset ISIC2019 --data_path ./data/ISIC2019/ --batch_size 64 --lr 0.002 --epochs 100 --gpu 2 --model_path ./results/19_/pre
```
Stage2: introduce the number of class difficulty-dependent proxies 
```python
python train.py --dataset ISIC2019 --data_path ./data/ISIC2019/ --batch_size 64 --lr 0.002 --epochs 100 --gpu 2 --model_path ./results/19_/tr
```


## Citation & Acknowledgments

If you find this repo useful for your research, please consider citing the paper
