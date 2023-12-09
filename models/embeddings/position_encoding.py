import torch
import torch.nn as nn
import numpy as np



class PositionEncoding(nn.Module):
    def __init__(self,max_len, d_model, device):
        super(PositionEncoding,self).__init__()

        self.pos_code=torch.zeros(max_len, d_model,device=device)
        self.pos_code.requires_grad=False

        pos=torch.tensor(np.arange(max_len),device=device).unsqueeze(dim=1)
        _2i=torch.tensor(np.arange(d_model,step=2),device=device)

        self.pos_code[:,::2]=torch.sin(pos/(10000**(_2i/d_model)))
        self.pos_code[:,1::2]=torch.cos(pos/(10000**(_2i/d_model)))
    
    def forward(self,x:torch.Tensor):
        bs, length=x.size()

        # pos_code 和bs没有关系，不同的bs位置编码都是一样的
        return self.pos_code[:length,:]