import torch
from grace.utils import param_subset, get_logits, brackets_to_periods, parent_module
import transformers

class MemoryNetwork:
    def __init__(self, config, model):
        self.model = model.model
        self.tokenizer = model.tokenizer
        layer = config["model"]["inner_params"][0]
        self.config = config
        self.device = config["device"]
        self.nkeys = config.editor.nkeys
        
        for p in self.model.parameters():
            p.requires_grad = False
            
        # --- ensure proper formatting (GRACE edits ~layers~ not weights matrices) ---        
        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer
            
        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True
            
        # --- Add adaptors to chosen layers ---            
        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        layer_name = self.layer.rsplit(".", 1)[-1]
        original_layer = getattr(edit_module, layer_name)
        setattr(edit_module, layer_name, MemoryAdaptor(config, original_layer, transpose=transpose).to(self.device))
                
    def __call__(self, **kwargs):
        if self.config["experiment"]["task"] == "hallucination":
            key_id = (kwargs["labels"] == -100).sum() - 1
            setattr(eval(f"self.model.{self.layer}"), "key_id", key_id) # Tell GRACE which token to use for its query (default is the last token)
        return self.model(**kwargs)
    
    def edit(self, config, tokens, batch_history):
        setattr(eval(f"self.model.{self.layer}"), "train", True)
        optimizer = torch.optim.Adam(self.model.parameters(), config.editor.edit_lr)
        self.losses = []
        
        # Tell the model which token to replace
        if config["experiment"]["task"] == "hallucination":
            key_id = (tokens["labels"] == -100).sum() - 1
            setattr(eval(f"self.model.{self.layer}"), "key_id", key_id)
            
        for i in range(config.editor.n_iter):
            outputs = self.model(**tokens)
            loss = outputs.loss
            self.losses.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        self.loss = loss
        return self.model

class MemoryAdaptor(torch.nn.Module):
    def __init__(self, config, model, transpose):
        super(MemoryAdaptor, self).__init__()
        
        self.model = model
        self.device = model.weight.device
        
        if transpose:
            self.key_shape = model.weight.shape[1]
            self.value_shape = model.weight.shape[0]               
        else:
            self.key_shape = model.weight.shape[0]
            self.value_shape = model.weight.shape[1]
        nkeys = config["editor"]["nkeys"]
        self.fc = torch.nn.Linear(self.key_shape, nkeys).to(self.device)
        self.values = torch.nn.Parameter(torch.rand((nkeys, self.value_shape), requires_grad=True, device=self.device))
        self.key_id = -1
    
    def forward(self, *args):
        # Pull out query for current instance (activation computed by previous layer) 
        query = args[0][:, self.key_id, :] # Just use activation for last token
        key_weights = torch.softmax(self.fc(query), 1)
        value = (key_weights.view(-1, 1) * self.values).sum(0)
        return value.unsqueeze(0)
