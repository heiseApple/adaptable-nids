import torch
from torch import nn
from torchmetrics import Accuracy

from module.gradient_reverse_function import WarmStartGradientReverseLayer


class DistillKLLoss(nn.Module):
    """
    Implementation of the Kullback-Leibler divergence for distilliation
    """
    def __init__(self, T):
        super(DistillKLLoss, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, already_soft_values=False):
        if y_t is None:
            return 0.0

        p_s = nn.functional.log_softmax(y_s / self.T, dim=1)
        p_t = nn.functional.softmax(y_t / self.T, dim=1) if not already_soft_values else y_t
        loss = nn.functional.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.size(0)
        return loss
    
    
class DomainAdversarialLoss(nn.Module):
    """
    The Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_
    [[Link to Source Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/alignment/dann.py)
    """
    def __init__(self, domain_discriminator, reduction='mean', grl=None, sigmoid=True):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = grl or WarmStartGradientReverseLayer(
            alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True
        ) 
        self.domain_discriminator = domain_discriminator
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.domain_discriminator_accuracy = None

    def forward(self, f_s, f_t, w_s=None, w_t=None):
        # Concatenate source and target features, then apply gradient reversal
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        # Pass concatenated features through the domain discriminator
        d = self.domain_discriminator(f)
        
        batch_size_s, batch_size_t = f_s.size(0), f_t.size(0)
        
        if self.sigmoid:
            bin_accuracy = Accuracy(task='binary').to(f_s.device)
            # Split the discriminator output back into source/target
            d_s, d_t = torch.split(d, [batch_size_s, batch_size_t], dim=0)
            
            # Construct labels: source=1, target=0
            d_label_s = torch.ones(batch_size_s, 1, device=f_s.device)
            d_label_t = torch.zeros(batch_size_t, 1, device=f_t.device)
            
            # Calculate domain discriminator accuracy for monitoring
            acc_s = bin_accuracy(d_s, d_label_s).item()
            acc_t = bin_accuracy(d_t, d_label_t).item()
            self.domain_discriminator_accuracy = 0.5 * (acc_s + acc_t)

            # Set default weights if None
            w_s = torch.ones_like(d_label_s) if w_s is None else w_s.view_as(d_s)
            w_t = torch.ones_like(d_label_t) if w_t is None else w_t.view_as(d_t)
                
            # Compute BCE losses for source and target, then average
            loss_s = nn.functional.binary_cross_entropy(
                d_s, d_label_s, weight=w_s, reduction=self.reduction)
            loss_t = nn.functional.binary_cross_entropy(
                d_t, d_label_t, weight=w_t, reduction=self.reduction)
            return 0.5 * (loss_s + loss_t)
            
        else:
            accuracy = Accuracy(num_classes=2, task='multiclass').to(f_s.device)
            # Create integer labels: source=1, target=0
            d_label = torch.cat([
                torch.ones(batch_size_s, device=f_s.device, dtype=torch.long),
                torch.zeros(batch_size_t, device=f_t.device, dtype=torch.long)
            ])
            
            # Compute accuracy for monitoring
            self.domain_discriminator_accuracy = accuracy(d, d_label).item()
            
            # Set default weights if None
            w_s = torch.ones(batch_size_s, device=f_s.device) if w_s is None else w_s
            w_t = torch.ones(batch_size_t, device=f_t.device) if w_t is None else w_t
            weights = torch.cat([w_s, w_t])
                            
            # Calculate weighted cross-entropy loss
            ce_loss = nn.functional.cross_entropy(d, d_label, reduction='none')
            weighted_loss = ce_loss * weights

            if self.reduction == 'mean':
                return weighted_loss.mean()
            elif self.reduction == 'sum':
                return weighted_loss.sum()
            elif self.reduction == 'none':
                return weighted_loss
            else:
                raise ValueError(f'Unsupported reduction mode: {self.reduction}')
            