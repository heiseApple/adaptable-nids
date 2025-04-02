import numpy as np
import torch
from torch import nn
from torch.autograd import Function


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, input, coeff):
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    """
    A PyTorch module that implements a warm-start gradient reversal layer. This layer reverses 
    the gradient during backpropagation with a coefficient that changes dynamically based on 
    the training iteration. It is commonly used in domain adaptation tasks.
    [[Link to Source Code]](https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/modules/grl.py)
    
    Args:
        alpha (float): Scaling factor for the gradient reversal coefficient. Default is 1.0.
        lo (float): Minimum value of the coefficient. Default is 0.0.
        hi (float): Maximum value of the coefficient. Default is 1.0.
        max_iters (int): Maximum number of iterations for coefficient adjustment. Default is 1000.
        auto_step (bool): Whether to automatically increment the iteration counter. Default is False.
    """
    def __init__(self, alpha, lo, hi, max_iters, auto_step):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        self.iter_num += 1
