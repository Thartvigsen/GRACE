#!/bin/bash

# Examples
# SCOTUS experiment
python ./grace/main.py experiment=scotus model=scotus-bert editor=grace

# Hallucination experiment (requires finetuning GPT2-XL model!)
# python ./grace/main.py experiment=hallucination model=gpt2xl editor=grace

# QA experiment (requires dataset downloads to use!)
# python ./grace/main.py experiment=qa model=t5small editor=grace
