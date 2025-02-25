import torch
from torch import nn
from tqdm import tqdm

from approach.dl_module import DLModule
from callback.freeze_backbone import FreezeBackbone
from module.head import NNHead
from module.loss import DistillKLLoss
from module.teacher import RFSTeacher
from util.config import load_config


class RFS(DLModule):
    """
    [[Link to Source Code]](https://github.com/RL-VIG/LibFewShot)
    RFS is a class that implements the RFS (Rethinking Few-Shot) algorithm for few-shot learning,
    as described in "Rethinking Few-Shot Image Classification: a Good Embedding Is All You Need?".
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cf = load_config()
        
        
        self.kd_T = kwargs.get('kd_t', cf['kd_t'])
        self.is_distill = kwargs.get('is_distill', cf['is_distill'])
        self.teacher_path = kwargs.get('teacher_path', cf['tacher_path'])
        self.alpha = kwargs.get('alpha', cf['alpha']) if self.is_distill else 0
        self.gamma = kwargs.get('gamma', cf['gamma']) if self.is_distill else 1
        if self.alpha + self.gamma != 1.0:
            raise ValueError('alpha + gamma should be equal to 1')

        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self.kl_loss = DistillKLLoss(T=self.kd_T)
        
        self.teacher = RFSTeacher(
            net=self.net, 
            is_distill=self.is_distill, 
            teacher_path=self.teacher_path
        )
        # RFS freezes the backbone during the adaptation phase
        self.adaptation_strat = 'freezing'
        self.callbacks.append(FreezeBackbone())
        
        
    @staticmethod
    def add_appr_specific_args(parent_parser):
        cf = load_config()
        parser = DLModule.add_appr_specific_args(parent_parser)
        parser.add_argument('--alpha', type=float, default=cf['alpha'])
        parser.add_argument('--gamma', type=float, default=cf['gamma'])
        parser.add_argument('--is-distill', action='store_true', default=cf['is_distill'])
        parser.add_argument('--kd-t', type=float, default=cf['kd_t'])
        parser.add_argument('--teacher-path', type=str, default=cf['tacher_path'])
        return parser
    
    
    def _fit_step(self, batch_x, batch_y):
        student_logits = self.net(batch_x)
        teacher_logits = self.teacher(batch_x)
        
        # CE on actual label and student logits
        gamma_loss = self.ce_loss(student_logits, batch_y)
        # KL on teacher and student logits
        alpha_loss = self.kl_loss(student_logits, teacher_logits)
        
        loss = gamma_loss * self.gamma + alpha_loss * self.alpha
        return loss, student_logits
                
            
    def _predict_step(self, batch_x, batch_y):
        logits = self.net(batch_x)
        loss = self.ce_loss(logits, batch_y)
        return loss, logits
        
        
    @torch.no_grad()
    def _adapt(self, train_dataloader, _):
        self.net.eval()
        
        embeddings, labels = [], []
        
        for batch_x, batch_y in tqdm(train_dataloader, desc='[fitting NN head]', leave=True):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # Embed the input
            _, batch_emb = self.net(batch_x, return_feat=True)
            embeddings.append(batch_emb)
            labels.append(batch_y)
            
        # Replace the FullyConnected head with a nearest neighbor 
        self.net.set_head(NNHead())
        # Store train embedding and labels
        self.net.head.fit(x=torch.cat(embeddings), y=torch.cat(labels))
        