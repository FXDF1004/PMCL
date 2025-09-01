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
![Our Network Structure](network.png)
Skin image datasets often suffer from imbalanced data distribution, exacerbating the difficulty of computer-aided skin disease diagnosis.  Some recent works exploit supervised contrastive learning (SCL) for this long-tailed challenge. Despite achieving significant performance, these SCL-based methods focus more on head classes, yet ignore the utilization of information in tail classes. In this paper, we propose class-*E*nhancement *C*ontrastive *L*earning (*ECL*), which enriches the information of minority classes and treats different classes equally. For information enhancement, we design a hybrid-proxy model to generate class-dependent proxies and propose a cycle update strategy for parameter optimization. A balanced-hybrid-proxy loss is designed to exploit relations between samples and proxies with different classes treated equally. Taking both "imbalanced data" and "imbalanced diagnosis difficulty" into account, we further present a balanced-weighted cross-entropy loss following the curriculum learning schedule.

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
