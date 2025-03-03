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



---
![overall_structure](./datafree/pipeline.jpg)

---

## Installation & Preparation

1. Clone the repo and prepare the virtual environment.

```
git clone https://github.com/LiQiiiii/Encapsulating-Knowledge-In-One-Prompt.git
```

```
cd Encapsulating-Knowledge-In-One-Prompt
```

```
conda create -n kiop python=3.10.0
```

```
conda activate kiop
```

```
pip install -r requirements.txt
```

2. Prepare the dataset and models. You can use your own models and dataset. For quick start, we provide several models and datasets, which can be downloaded directly from google drive:

```
gdown https://drive.google.com/uc?id=19o2EItRw-LOJUdjDf-mOz0zh0QalF8wj
```

```
gdown https://drive.google.com/uc?id=18XDK2fdhCQuwGm4sJntfSvESpbZEv1bY
```

```
unzip KiOP_models.zip
```

```
unzip KiOP_data.zip
```


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


