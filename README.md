# Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adapters
This repository contains the code to run GRACE: General Retrieval Adapters for Continual Editing, proposed in the paper **[Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adapters](https://arxiv.org/abs/2211.11031)**.

<img width="1866" alt="image" src="https://github.com/Thartvigsen/GRACE/assets/26936677/8f28ab99-2411-4fd8-949b-8373ebfff3b5">

Please feel free to email me at `tomh@mit.edu` or raise an issue with this repository and I'll get back to you as soon as I can.

## Setup
1. Create a virtual environment (we use conda)
    ```
    conda env create --name grace_env --file environment.yml
    ```
2. Activate the virtual environment
    ```
    conda activate grace_env
    ```
3. Install the repository
    ```
    pip install -e .
    ```

## Data
The QA experiments use data linked by the [MEND](https://github.com/eric-mitchell/mend) repository. Per their instructions, you can download the data for NQ and zsRE from [their Google Drive link](https://drive.google.com/drive/folders/1jAqBE45jEKR-5pMkwxlVQ0V8eKxqWbxA) and unzip each sub-directory into `grace/data`. SCOTUS and Hallucination data are handled through huggingface.

## Running experiments
Experiments are run using [main.py](./grace/main.py). Experiment settings and hyperparameters are chosen using [hydra](https://github.com/facebookresearch/hydra). While more examples are available in [./scripts/main.sh](./scripts/main.sh), three representative experiments can be run as follows:

### Editing GPT2-XL on Hallucination with GRACE
```
python grace/main.py experiment=hallucination model=gpt2xl editor=grace
```

### Editing BERT on SCOTUS with GRACE
```
python grace/main.py experiment=scotus model=bert editor=grace
```

### Editing T5 on zsRE with GRACE
```
python grace/main.py experiment=qa model=t5small editor=grace
```

## Repository Roadmap
* [./scripts/](./scripts/) contains handy shell scripts for starting and running experiments in slurm.
* [./notebooks/](./notebooks/) contains a simple example of editing a model with GRACE.
* [./ckpts/](./ckpts/) will contain checkpoints of your edited models if you choose to checkpoint models.
* [./data/](./data/) will contain downloaded datasets if you choose to cache data yourself instead of relying on HuggingFace.
* [./grace/](./grace/) contains the source code to GRACE
    * [./grace/main.py](./grace/main.py) is the main file to kick off experiments.
    * [./grace/config/](./grace/config/) contains the config files for datasets, editors, and pretrained models.
    * [./grace/editors/](./grace/editors/) contains source code for each compared editor.
    * [./grace/dataset.py](./grace/dataset.py) contains source code for each compared dataset.
    * [./grace/metrics.py](./grace/metrics.py) contains source code for each compared dataset.
    * [./grace/models.py](./grace/models.py) contains source code for loading pretrained models.

## Citation
Please use the following to cite this work:
```
@article{hartvigsen2023aging,
  title={Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adapters},
  author={Hartvigsen, Thomas and Sankaranarayanan, Swami and Palangi, Hamid and Kim, Yoon and Ghassemi, Marzyeh},
  journal={arXiv preprint arXiv:2211.11031},
  year={2023}
}
```
