import torch
from grace.utils import param_subset, get_logits, brackets_to_periods

class Finetune_retrain(torch.nn.Module):
    def __init__(self, config, model):
        """
        This method directly finetunes chosen weights given new inputs
        """
        super(Finetune_retrain, self).__init__()
        self.model = model.model
        self.tokenizer = model.tokenizer
        self.pnames = brackets_to_periods(config["model"]["inner_params"][0])
        self.device = config["device"]
        self.edit_lr = config.editor.edit_lr
        self.retrain_memory = config.editor.retrain_memory
        
        for n, p in self.model.named_parameters():
            if n != self.pnames:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def retrain(self, init_model, config, batch_history):
        params = param_subset(self.model.named_parameters(), [self.pnames])
        opt = torch.optim.Adam(params, lr=self.edit_lr)
        for _ in range(5):
            for tokens in batch_history:
                self.model.zero_grad()
                outputs = self.model(**tokens)
                logits, loss = outputs.logits, outputs.loss
                loss.backward()
                opt.step()
                opt.zero_grad()
        return init_model
        
    def edit(self, config, tokens, *args):
        params = param_subset(self.model.named_parameters(), [self.pnames])
        opt = torch.optim.Adam(params, lr=self.edit_lr)
        self.losses = []
        for _ in range(config.n_iter):
            self.model.zero_grad()
            outputs = self.model(**tokens)
            logits, loss1 = outputs.logits, outputs.loss
            argmaxs = torch.argmax(logits, dim=-1)
            response_indices = (tokens['labels'] != -100)
            if torch.all(tokens['labels'][response_indices] == argmaxs[response_indices]).item():
                break
            self.loss = loss1
            self.losses.append(self.loss.detach().cpu().numpy())
            self.loss.backward()
            opt.step()
            opt.zero_grad()
        return self.model
