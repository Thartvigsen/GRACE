import torch
from grace.utils import param_subset, get_logits, brackets_to_periods

class Finetune_ewc(torch.nn.Module):
    def __init__(self, config, model):
        """
        This method directly finetunes chosen weights given new inputs
        """
        super(Finetune_ewc, self).__init__()
        self.model = model.model
        self.tokenizer = model.tokenizer
        self.pnames = brackets_to_periods(config["model"]["inner_params"][0])
        self.device = config["device"]
        self.ewc_lambda = config.editor.ewc_lambda
        self.fisher_mem = config.editor.fisher_mem
        self.edit_lr = config.editor.edit_lr
        
        for n, p in self.model.named_parameters():
            if n != self.pnames:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def compute_fisher_matrix(self, batch_history):
        optpar_dict = {}
        fisher_dict = {}
        model_dict = dict(self.model.named_parameters())
        for item_num, tokens in enumerate(batch_history[::-1]):
            if item_num < self.fisher_mem:
                outputs = self.model(**tokens)
                logits, loss = outputs.logits, outputs.loss
                loss.backward()

                for name in self.pnames:
                    if name not in optpar_dict:
                        optpar_dict[name] = model_dict[name].data.clone()
                        fisher_dict[name] = model_dict[name].grad.data.clone().pow(2)
                    else:
                        optpar_dict[name] += model_dict[name].data.clone()
                        fisher_dict[name] += model_dict[name].grad.data.clone().pow(2)
        for name in self.pnames:
            optpar_dict[name] /= self.fisher_mem
            fisher_dict[name] /= self.fisher_mem

        return fisher_dict, optpar_dict

    def edit(self, config, tokens, batch_history):
        params = param_subset(self.model.named_parameters(), self.pnames)
        opt = torch.optim.Adam(params, lr=self.edit_lr)
        self.losses = []
        fisher_dict, optpar_dict = self.compute_fisher_matrix(batch_history)
        for _ in range(config.n_iter):
            self.model.zero_grad()
            outputs = self.model(**tokens)
            logits, loss1 = outputs.logits, outputs.loss

            argmaxs = torch.argmax(logits, dim=-1)
            response_indices = (tokens['labels'] != -100)
            if torch.all(tokens['labels'][response_indices] == argmaxs[response_indices]).item():
                break

            # Add EWC regularization term
            for n, p in zip(self.pnames, params):
                ewc_regularizer = self.ewc_lambda * torch.sum(fisher_dict[n] * (p - optpar_dict[n]) ** 2)
                loss1 += ewc_regularizer

            self.loss = loss1
            self.losses.append(self.loss.detach().cpu().numpy())
            self.loss.backward()
            opt.step()
            opt.zero_grad()
        return self.model
