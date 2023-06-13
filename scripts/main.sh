#!/bin/bash
source ~/.bashrc
conda activate base

export XDG_RUNTIME_DIR=""

# Examples
# python ./grace/main.py experiment=scotus model=scotus-bert editor=grace

# python ./grace/main.py experiment=hallucination model=gpt2xl editor=grace

# python ./grace/main.py experiment=qa model=t5small editor=grace