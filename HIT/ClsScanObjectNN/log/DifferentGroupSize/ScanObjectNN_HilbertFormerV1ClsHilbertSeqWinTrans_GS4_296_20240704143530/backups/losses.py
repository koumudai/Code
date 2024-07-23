import torch
import torch.nn.functional as F


class SmoothCrossEntropy(torch.nn.Module):
    def __init__(self, use_smoothing=True, eps=0.2):
        super().__init__()
        self.use_smoothing = use_smoothing
        self.eps = eps if use_smoothing else 0
            
    def forward(self, pred, gold):
        if len(pred.shape) > 2:
            pred = pred.reshape(-1, pred.shape[-1])
        gold = gold.reshape(-1)

        if self.use_smoothing:
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
            loss = loss[torch.isfinite(loss)].mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss
