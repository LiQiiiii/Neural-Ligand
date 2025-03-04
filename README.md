<div align="center">
<h1><a href="https://arxiv.org/abs/2503.01268" target="_blank">Multi-Level Collaboration in Model Merging</a></h1>

<div>
<a target="_blank" href="https://arxiv.org/abs/2503.01268">
  <img src="https://img.shields.io/badge/arXiv-2503.01268-b31b1b.svg" alt="arXiv Paper"/>
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
<table>
    <tr>
        <td align="center">
            <img src="src/figures/neulig_overview.png" width="70%">
            <p><strong>Figure 1.</strong> An illustration of Portland, which consists of a linear layer followed by a softmax function.</p>
        </td>
        <td align="center">
            <img src="src/figures/neulig_train_pip.png" width="80%">
            <p><strong>Figure 2.</strong> The training process of Portland. </p>
        </td>
    </tr>
</table>

<table>

<div align="center">
    <div style="max-width: 30%; text-align: left; margin-bottom: 20px;">
        <img src="src/figures/exp.png" alt="Diagram 1" style="display: block; margin: 0 auto; width: 60%;">
    </div>
  </div>
<strong>Figure 3.</strong> A toy experiment to verify theoretical feasibility. In this experiment, we merged two models that were fine-tuned on different datasets. <strong>Marker shapes</strong> represent different methods, while <strong>colors</strong> indicate different experimental groups, with each group using a distinct combination of datasets. In total, 10 groups are conducted (represented by 10 different colors). <strong>Hollow markers</strong> for each method indicate the average results across these 10 groups.

<div align="center">
    <div style="max-width: 60%; text-align: left; margin-bottom: 20px;">
        <img src="src/figures/comparison.png" alt="Diagram 2" style="display: block; margin: 0 auto; width: 60%;">
    </div>
</div>
<strong>Table 1.</strong> The asterisk indicates that the condition is <strong>partially satisfied</strong>. For Simple-Averaging, the theoretical discussion is limited to the relationship between the performance of merging two models and that of ensembling. Furthermore, although both Simple-Averaging and Task-Arithmetic can be applied to CNN-based models, their performance is suboptimal. In the case of Diverse-Origin Models, all previous methods yield performance close to random guessing, but our conclusions remain applicable.
            
<div align="center">
    <div style="max-width: 100%; text-align: left; margin-bottom: 20px;">
        <img src="src/figures/main_table.png" alt="Diagram 3" style="display: block; margin: 0 auto; width: 100%;">
    </div>
</div>
<strong>Table 2.</strong> Results of various methods across multiple datasets, including the merging performance, the ensembling performance, and the performance gap for both CLIP-RN50 and CLIP-ViT-B/32.

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

2. Prepare the dataset and models. The download link of the datasets used in the paper can be found in `./data/data_links.txt`. Save them in the `./data` folder. Run:

```
python ./src/finetune_clean.py
```

to get the corresponding models for the training and evaluation.

---

## Training & Evaluation

```
python ./src/neulig_main.py --num_co_models 2 --global_epoch 1000 --alignment_type sup --model RN50
```

where `--num_co_models` is the number of collaborating models, `--alignment_type` controls the alignment term (i.e., sup/semi), and `--model` controls the model type (i.e., RN50/ViT-B-32/ViT-L-14).

If you finding our work interesting or helpful to you, please cite as follows:

```
@inproceedings{li2024encapsulating,
  title={Encapsulating Knowledge in One Prompt},
  author={Li, Qi and Yu, Runpeng and Wang, Xinchao},
  booktitle={European Conference on Computer Vision},
  pages={215--232},
  year={2024},
  organization={Springer}
}
```
