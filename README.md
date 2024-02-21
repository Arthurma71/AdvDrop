# AdvDrop


## Overview

Code of "General Debiasing for Graph-based Collaborative Filtering via Adversarial Graph Dropout"


## Run the Code

- We provide implementation for various baselines presented in the paper.

- To run the code, first run the following command to install tools used in evaluation:

```
python setup.py build_ext --inplace
```

### AdvDrop Training 

Coat:
```python
python run_advdrop.py --modeltype AdvDrop --dataset Coat.new --n_layers 2 --neg_sample 1 --saveID yourID --lr 1e-3 --interval 7 --adv_epochs 10  --batch_size 128 --adv_lr 0.01 --embed_size 30  
```

Yahoo:
```python
python run_advdrop.py --modeltype AdvDrop --dataset Yahoo.new --n_layers 2 --neg_sample 1 --saveID yourID --lr 3e-3 --interval 15 --adv_epochs 5  --batch_size 128 --adv_lr 0.001 --embed_size 30  
```

KuaiRec:
```python
python run_advdrop.py --modeltype AdvDrop --dataset KuaiRec.new --n_layers 2 --neg_sample 1 --saveID yourID --lr 5e-4 --interval 3 --adv_epochs 5  --batch_size 512 --adv_lr 0.001 --embed_size 30  
```

Yelp2018:
```python
python run_advdrop.py --modeltype AdvDrop --dataset Yelp2018.new --n_layers 2 --neg_sample 1 --saveID yourID --lr 5e-4 --interval 7 --adv_epochs 15  --batch_size 1024 --adv_lr 0.01 --embed_size 64
```

- Douban:

```python
python run_advdrop.py --modeltype AdvDrop --dataset Douban.new --n_layers 2 --neg_sample 1 --saveID yourID --lr 5e-4 --interval 10 --adv_epochs 3  --batch_size 4096 --adv_lr 0.01 --embed_size 64
```

### Other Baselines Training 

```python
python main.py --modeltype `Model` --dataset `Dataset` --n_layers 2 --neg_sample 1 --saveID yourID 
```
Please replace 'Model' with the baseline name and replace 'Dataset' with the name of the dataset you intend to evaluate. And don't forget to add specific params for the baselines.



## Requirements

- python == 3.7.10

- pytorch == 1.12.1+cu102

- tensorflow == 1.14

- reckit == 0.2.4

## Reference
If you want to use our codes and datasets in your research, please cite:

```
@inproceedings{AdvDrop,
  title={General Debiasing for Graph-based Collaborative Filtering via
Adversarial Graph Dropout},
  author={Zhang, An and Ma, Wenchang, and Wei, Pengbo, and Sheng, Leheng and Wang, Xiang},
  booktitle={{WWW}},
  year={2024}
}
```



