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
    xML-Lab, National University of Singapore&emsp;
    <sup>&dagger;</sup>corresponding author 
</div>
</div>
</div>

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

2. Prepare the dataset and models. The download link of the datasets used in the paper can be found in /data/data_links.txt.

---

## Training & Evaluation

We provide several scripts in ```./scripts```. For example, for running KiOP-B, you may use the ```KiOP_B.sh``` as follows. You can adjust the hyperparameters in the shell file to customize your setup:

```
sh ./scripts/KiOP_B.sh
```

## Citation

If you finding our work interesting or helpful to you, please cite as follows:

```
@misc{li2024encapsulatingknowledgeprompt,
      title={Encapsulating Knowledge in One Prompt}, 
      author={Qi Li and Runpeng Yu and Xinchao Wang},
      year={2024},
      eprint={2407.11902},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements

This implementation is built on top of the code from [ILM-VP](https://github.com/OPTML-Group/ILM-VP) and [CMI](https://github.com/zju-vipa/CMI). We would like to express our gratitude to the authors of these repositories.


