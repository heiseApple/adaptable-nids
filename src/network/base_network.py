import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from util.directory_manager import DirectoryManager


class BaseNetwork(nn.Module, ABC):
    
    def __init__(self):
        super(BaseNetwork, self).__init__()
        self.out_features_size = None
        self.backbone = None
        self.head = None

    @abstractmethod
    def forward(self, x, return_feat=False):
        """
        Forward pass of the network.
        """
        pass
    
    @abstractmethod
    def extract_features(self, x):
        """
        Extracts features from the backbone of the network.
        """
        pass

    def set_head(self, head: nn.Module):
        """
        Configures or replaces the classification head of the model.
        """
        self.head = head

    def freeze_backbone(self):
        """
        Freezes all the parameters in the backbone, preventing gradient updates.
        """
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreezes the parameters in the backbone, allowing gradient updates.
        """
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
    def freeze_net(self):
        """
        Freezes all the parameters, preventing gradient updates.
        """
        if self.backbone and self.head:
            for module in [self.backbone, self.head]:
                for param in module.parameters():
                    param.requires_grad = False

    def unfreeze_net(self):
        """
        Unfreezes the parameters, allowing gradient updates.
        """
        if self.backbone and self.head:
            for module in [self.backbone, self.head]:
                for param in module.parameters():
                    param.requires_grad = True
                    
    def freeze_bn(self):
        """
        Freeze all batch normalization layers in backbone and head.
        """
        if self.backbone and self.head:
            for module in (self.backbone, self.head):
                for m in module.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        m.eval()

    def unfreeze_bn(self):
        """
        Unfreeze all batch normalization layers in backbone and head.
        """
        if self.backbone and self.head:
            for module in (self.backbone, self.head):
                for m in module.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        m.train()
                        
    def save_weights(self, filename):
        """
        Saves the model weights to a specified file path.
        """
        # for name, param in self.named_parameters():
        #     print(name, param.device, torch.sum(param).item())
        dm = DirectoryManager()
        path = dm.mkdir('network_weights')
        weights_path = f'{path}/{filename}.pt'
        torch.save({'net_state_dict' : self.state_dict()}, weights_path)
        return weights_path

    def load_weights(self, path):
        """
        Loads the model weights from a specified file path.
        """
        if path is None:
            raise ValueError('The path to the weights file must be specified.')
        self.load_state_dict(torch.load(path)['net_state_dict'])
        # for name, param in self.named_parameters():
        #     print(name, param.device, torch.sum(param).item())
        
    def trainability_info(self):
        """
        Prints the trainability status of each parameter in the network.
        """
        print('[Trainability Information]')
        print('-'*80)
        print(f"{'Layer (type/param)':<50} | {'Requires Grad'}")
        print('-'*80)
        for name, param in self.named_parameters():
            print(f'{name:<50} | {param.requires_grad}')
        print('-'*80)
        
    def summarize_module(self, print_fn=print):
        """
        Prints a summary of the module's layers, their shapes, and the number of parameters.
        """
        print_fn('-'*80)
        print_fn(f"{'Layer (type/param)':<35} | {'Shape':<20} | # Params")
        print_fn('-'*80)
        
        total_params = 0
        trainable_params = 0

        for name, param in self.named_parameters():
            param_count = param.numel()
            
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
            
            print_fn(f'{name:<35} | {str(list(param.shape)):<20} | {param_count}')

        non_trainable_params = total_params - trainable_params

        print_fn('-'*80)
        print_fn(f'Total params:         {total_params}')
        print_fn(f'Trainable params:     {trainable_params}')
        print_fn(f'Non-trainable params: {non_trainable_params}')
        print_fn('-'*80)

    def _get_padding(self, kernel, padding='same'):
        # http://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
        pad = kernel - 1
        if padding == 'same':
            if kernel % 2:
                return pad // 2, pad // 2
            else:
                return pad // 2, pad // 2 + 1
        return 0, 0

    def _get_output_dim(self, dimension, kernels, strides, padding='same', return_paddings=False):
        # http://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
        out_dim = dimension
        paddings = []
        if padding == 'same':
            for kernel, stride in zip(kernels, strides):
                paddings.append(self._get_padding(kernel, padding))
                out_dim = (out_dim + stride - 1) // stride
        else:
            for kernel, stride in zip(kernels, strides):
                paddings.append(self._get_padding(kernel, padding))
                out_dim = (out_dim - kernel + stride) // stride

        if return_paddings:
            return out_dim, paddings
        return out_dim