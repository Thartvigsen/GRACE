import torch
import logging
import transformers
from torch import nn
from grace.utils import ckpt_dir

LOG = logging.getLogger(__name__)

def pretrain(model, loader, tokenize, n_epochs, device):
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
    for _ in range(n_epochs):
        losses = []
        for batch in loader:
            batch = tokenize(batch, model.tokenizer, device)
            loss = model.model(**batch).loss
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.model.zero_grad()
    return model

def get_tokenizer(config):
    tok_name = config.model.tokenizer_name if config.model.tokenizer_name is not None else config.model.name
    return getattr(transformers, config.model.tokenizer_class).from_pretrained(tok_name, cache_dir=ckpt_dir())

def get_hf_model(config):
    ModelClass = getattr(transformers, config.model.class_name)
    LOG.info(f"Loading model class {ModelClass} with name {config.model.name} from cache dir {ckpt_dir()}")
    if config.model.pt is None:    
        model = ModelClass.from_pretrained(config.model.name, cache_dir=ckpt_dir())
    elif config.re_init_model:
        print("Downloading untrained model.")
        model = ModelClass.from_pretrained(config.model.name)
    else:
        try:
            # try to load specified model from local dir
            model = ModelClass.from_pretrained(config.model.pt)
            print(f"Loaded model: {config.model.pt}")
        except:
            print("Couldn't load model: {config.model.pt}. Downloading new model.")
            model = ModelClass.from_pretrained(config.model.name, cache_dir=ckpt_dir())

    if config.dropout is not None:
        n_reset = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = config.dropout
                n_reset += 1
    return model

class Classifier(torch.nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.model = get_hf_model(config)
        self.tokenizer = get_tokenizer(config)

    def forward(self, tokens):
        output = self.model(**tokens)
        logits = output.logits
        loss = torch.nn.functional.cross_entropy(logits, tokens["labels"])
        out = {
            "logits": logits,
            "loss": loss
        }
        return out

class QAModel(torch.nn.Module):
    def __init__(self, config):
        super(QAModel, self).__init__()
        self.model = get_hf_model(config).eval()
        self.tokenizer = get_tokenizer(config)
        self.device = config["device"]

    def forward(self, batch):
        logits = []
        self.loss = []
        for item in batch["text"]:
            item = {f"{k1}" : v1.to(self.device) for k1, v1 in item.items()}
            output = self.model(**item)
            logits.append(output.logits)
            try:
                self.loss.append(output.loss)
            except:
                pass
        self.loss = torch.stack(self.loss).mean()
        return torch.stack(logits)

    def get_loss(self, logits, batch):
        return self.loss

class GPTModel(torch.nn.Module):
    def __init__(self, config):
        super(GPTModel, self).__init__()
        self.model = get_hf_model(config).eval()
        self.device = config.device
        self.tokenizer = get_tokenizer(config)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = config["device"]

    def forward(self, batch):
        logits = self.model(**batch)
        return logits

    def get_loss(self, model, batch):
        loss = model(**batch).loss
        return loss
