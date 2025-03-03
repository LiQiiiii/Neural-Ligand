<div align="center">
<h1>Multi-Level Collaboration in Model Merging</h1>

<div>
<a target="_blank" href="https://arxiv.org/abs/***">
  <img src="https://img.shields.io/badge/arXiv-2312.17142-b31b1b.svg" alt="arXiv Paper"/>
</a>
</div>

<div>
Qi Li&emsp;Runpeng Yu&emsp;Xinchao Wang<sup>&dagger;</sup>
</div>
<div>
    <a href="https://sites.google.com/view/xml-nus/people?authuser=0" target="_blank">xML-Lab</a>, National University of Singapore&emsp;
    <sup>&dagger;</sup>corresponding author 
</div>
</div>
</div>

## Graphical Abstract
<table align="center">
    <tr>
        <td align="center">
            <img src="src/figures/neulig_overview.png" width="70%">
            <p>Figure 1. An illustration of Portland, which consists of a linear layer followed by a softmax function.</p>
        </td>
        <td align="center">
            <img src="src/figures/neulig_train_pip.png" width="70%">
            <p>Figure 2. The training process of Portland. </p>
        </td>
    </tr>
</table>


## Installation & Preparation

1. Clone the repo and prepare the virtual environment.

```
git clone https://github.com/LiQiiiii/Neural-Ligand.git
```

```
cd Neural-Ligand
```

```
conda create -n neulig python=3.8.10
```

```
conda activate neulig
```

The codes are tested on torch 2.0.0 and torchvision 0.15.1.

2. Prepare the dataset and models. The download link of the datasets used in the paper can be found in /data/data_links.txt. Run:

```
python ./src/finetune_clean.py
```

to get the corresponding models for the training and evaluation.

---

## Training & Evaluation

```
python ./src/neulig_main.py --num_co_models 2 --global_epoch 100 --alignment_type sup --model RN50
```

where `--num_co_models` is the number of collaborating models, `--alignment_type` controls the alignment term (i.e., sup/semi), and `--model` controls the model type (i.e., RN50/ViT-B-32/ViT-L-14).
