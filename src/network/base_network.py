import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from util.directory_manager import DirectoryManager


class BaseNetwork(nn.Module, ABC):
    
    def __init__(self):
        super(BaseNetwork, self).__init__()
        self.backbone = None
        self.head = None

    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the network.
        """
        # TODO: if return_feat is True, return the features extracted from the backbone
        pass
    
    @abstractmethod
    def extract_features(self, x):
        """
        Extracts features from the backbone of the network.
        """
        pass

    @abstractmethod
    def set_head(self, num_classes, out_features_size=None):
        """
        Configures or replaces the classification head of the model.
        """
        pass

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
                param.requires_grad = True

    def save_weights(self, filename):
        """
        Saves the model weights to a specified file path.
        """
        # for name, param in self.named_parameters():
        #     print(name, param.device, torch.sum(param).item())
        dm = DirectoryManager()
        path = dm.mkdir('network_weights')
        weights_path = f'{path}/{filename}.pt'
        dm.checkpoint_path = weights_path
        torch.save(self.state_dict(), weights_path)
        return weights_path

    def load_weights(self, path=None):
        """
        Loads the model weights from a specified file path.
        """
        self.load_state_dict(torch.load(path))
        # for name, param in self.named_parameters():
        #     print(name, param.device, torch.sum(param).item())
        
    def trainability_info(self):
        print('\nTRAINABILITY INFO')
        for name, param in self.named_parameters():
            print(name, param.requires_grad)
        print('')
        
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