import hydra
import copy
import logging
from tqdm import tqdm
from time import time
import wandb
import numpy as np
from grace.dataset import Hallucination, NQ, zsRE, SCOTUS, WebText10k
from grace.editors import GRACE, Finetune, Finetune_ewc, Finetune_retrain, Defer, MemoryNetwork
from grace.utils import *
from grace.metrics import F1, PPL, Accuracy, is_qa_error, is_acc_error
from grace.models import QAModel, Classifier, GPTModel, pretrain
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

OmegaConf.register_new_resolver("uuid", lambda: 1)

logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
LOG = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config", version_base="1.2")
def run(config: DictConfig):
    np.random.seed(42)
    
    if config.wandb:
        wandb.init(project=config.wandb_project_name, config=config, mode=config.wandb_mode)
        if not config.wandb_run_name:
            # If no specific run name is provided, try using the SLURM_JOBID. This only works if using SLURM, otherwise it'll just use default
            try:
                wandb.run.name = f"{str(config.editor._name)}_{os.getenv('SLURM_JOBID')}"
            except:
                pass
        wandb.run.save()
        wandb.config = config
    LOG.info(f"\n\n{OmegaConf.to_yaml(config)}\n")
    base_dir = hydra.utils.get_original_cwd()
    LOG.info(f"Project base directory: {base_dir}")
    
    ckpt_dir = config.ckpt_dir

    print(config)

    # --- Load Dataset and Model ---
    if config.experiment.task == "qa":
        # QA TASK
        model = QAModel(config).to(device)
        upstream = NQ()
        edits = zsRE(split="edit")
        edit_holdouts = zsRE(split="holdout")

        # --- get loaders ---
        edit_loader = DataLoader(edits, batch_size=1, shuffle=True)
        edit_holdout_loader = DataLoader(edit_holdouts, batch_size=1, shuffle=False)
        upstream_loader = DataLoader(upstream, batch_size=config.batch_size, shuffle=False)
        holdout = 0
        
        # --- define task-specific functions ---
        metric = F1 # Measure QA F1
        is_error = is_qa_error
        tokenize = tokenize_qa

    elif config.experiment.task == "scotus":
        # SCOTUS TASK
        model = Classifier(config).to(device)
        upstream = SCOTUS(split="train")
        edits = SCOTUS(split="edit")
        
        # --- get loaders ---
        edit_loader = DataLoader(edits, batch_size=1, shuffle=False)
        upstream_loader = DataLoader(upstream, batch_size=config.batch_size, shuffle=False)
        
        # --- define task-specific functions ---
        metric = Accuracy
        is_error = is_acc_error
        tokenize = tokenize_clf

    elif config.experiment.task == "hallucination":
        # HALLUCINATION TASK
        model = GPTModel(config).to(device)
        if config.pretrain or not config.model.pt:
            # --- pretrain our own model if desired or if no checkpoint already exists ---
            print("Pretraining model")
            pretrain_dataset = Hallucination(split="pretrain")
            pretrain_loader = DataLoader(pretrain_dataset, batch_size=2, shuffle=True)
            cache = os.path.join(ckpt_dir, "hallucination/")
            os.makedirs(cache, exist_ok=True)
            model = pretrain(model, pretrain_loader, n_epochs=1, device=device)
            model.model.save_pretrained(config.model.pt, from_pt=True)
            print("Model is trained and saved")

        # --- load model's training data, a new editing dataset, and pack them into dataloaders --- 
        edits = Hallucination(split="edit")
        accurate_dataset = Hallucination(split="accurate")
        upstream = WebText10k()
        edit_loader = DataLoader(edits, batch_size=1, shuffle=False)
        accurate_loader = DataLoader(accurate_dataset, batch_size=config.batch_size, shuffle=False)
        upstream_loader = DataLoader(upstream, batch_size=config.batch_size, shuffle=False)
        
        # --- define task-specific functions ---
        metric = PPL # Perplexity
        is_error = lambda *args: True # For language modeling, we've precomputed which inputs we count as "errors". In principle, this could be thresholded PPL or something.
        tokenize = tokenize_gpt
                
    else:
        print(f"{config.experiment.task} task not found")
        
    print(f"Loaded {len(edit_loader)} candidate edits.")

    # --- load editor ---
    if config.editor._name == "ft":
        editor = Finetune(config, model)
    elif config.editor._name == "ft_ewc":
        editor = Finetune_ewc(config, model)
    elif config.editor._name == "ft_retrain":
        editor = Finetune_retrain(config, model)
        unedited_model = copy.deepcopy(model)
    elif config.editor._name == "grace":
        editor = GRACE(config, model)
    elif config.editor._name == "memory":
        editor = MemoryNetwork(config, model)
    elif config.editor._name == "defer":
        editor = Defer(config, model)
    else:
        print(f"No editor class associated with {config.editor._name}")

    # --- let editor inherit .generate() ---
    editor.generate = model.model.generate
    
    # --- calculate pre-editing model behavior ---
    with torch.no_grad():
        original_edits = torch.tensor([metric(editor, tokenize(e, editor.tokenizer, config["device"])) for e in iter(edit_loader)]).nanmean() # Log first PPL before edits
        print("Average performance on edit set: ", original_edits)
        if config.experiment.task == "hallucination":
            ARR = torch.tensor([metric(editor, tokenize(e, editor.tokenizer, config["device"], test=True)) for e in iter(accurate_loader)]).nanmean() # Log first PPL before edits
            print("Original Accurate: ", ARR)
        TRR = torch.tensor([metric(editor, tokenize(e, editor.tokenizer, config["device"], test=True)) for e in iter(upstream_loader)])
        TRR = TRR[~torch.isnan(TRR)]
        TRR = TRR.nanmean()
        print("Original TRR: ", TRR)

    # --- begin editing ---
    n_edits = 0
    batch_history = []
    for i, batch in tqdm(enumerate(edit_loader)):
        tokens = tokenize(batch, editor.tokenizer, config["device"])
        
        print(i)
        print(is_error(editor, tokens))
        
        # --- Check that the model is actually making a mistake (for detecting hallucination, `is_error` always returns True) or stop after making enough edits ---
        if is_error(editor, tokens) & (n_edits <= config.max_n_edits):
            n_edits += 1
                
            batch_history.append(tokens) # Append new batch to growing history of edits

            # --- for methods we retrain, do that here every `retrain_frequency` steps after step 0 ---
            if "retrain" in config.editor._name and (i > 0 and n_edits % config.editor.retrain_frequency == 0):
                retrain_start =  time()
                editor.retrain(config=config, batch_history=batch_history)
                print(f'Retraining time: {time() - retrain_start}')

            # --- perform edit ---
            edit_start = time()
            editor.edit(config, tokens, batch_history)
            edit_time = time() - edit_start

            # --- Compute and log metrics ---
            log_dict = {}
            with torch.no_grad():
                ES = metric(editor, tokens)
                if i == 0:
                    ERR = ES

                if (i > 0 and n_edits % config.metric_period == 0) or (i == len(edit_loader)-1): # Compute historical metrics every k edits to save time
                    if config.experiment.task == "hallucination":
                        ARR = torch.tensor([metric(editor, tokenize(e, editor.tokenizer, config["device"])) for e in iter(accurate_loader)]).nanmean()
                    elif config.experiment.task == "qa":
                        holdout = torch.tensor([metric(editor, tokenize(e, editor.tokenizer, config["device"])) for e in iter(edit_holdout_loader)]).nanmean()
                    else:
                        pass
                    
                    ERR = torch.tensor([metric(editor, tokens) for tokens in batch_history]).nanmean()
                    TRR = torch.tensor([metric(editor, tokenize(e, editor.tokenizer, config["device"], test=True)) for e in iter(upstream_loader)])
                    TRR = TRR[~torch.isnan(TRR)] # Drop nans
                    TRR = torch.mean(TRR.nanmean()).squeeze()
             
            # --- Log metrics and push to Weights & Biases ---
            log_dict["TRR"] = TRR # Test Retention Rate
            log_dict["ERR"] = ERR # Error Retention Rate
            log_dict["ES"] =  ES # Edit Success
            log_dict["train_time"] = edit_time/60 # Time it takes to make one edit
            log_dict["edit"] = batch["text"] # Raw edit input
            log_dict["edit_label"] = batch["labels"] # Raw edit label
            log_dict["n_edits"] = n_edits # Raw edit label
            if config.experiment.task == "hallucination":
                log_dict["ARR"] = ARR # Accurate Retention Rate
            elif config.experiment.task == "qa":
                log_dict["Holdout"] = holdout
            else:
                pass
                
            if hasattr(editor, "log_dict"):
                log_dict.update(editor.log_dict) # Add any logged values inside our editor to log_dict
                
            print(log_dict)
            assert 2 == 3

            # --- if using weights and biases, upload the log ---
            if config.wandb:
                wandb.log(log_dict, step=i)
            else:
                # Print all logged vals
                print(f"Number of edits: {n_edits}")
                for k in log_dict:
                    print(f"{k}: {log_dict[k]}")

    ### --- save final edited model ---
    if config.ckpt:
        os.makedirs(os.path.join(ckpt_dir(), wandb.run.name), exist_ok=True)
        OmegaConf.save(config, os.path.join(ckpt_dir, "config.yaml"))
        torch.save(editor.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))

if __name__ == "__main__":
    run()
