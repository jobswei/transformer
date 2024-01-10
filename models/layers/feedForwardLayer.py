import torch
import torch.nn as nn

class FeedForwardLayer(nn.Module):
    def __init__(self,d_model,hidden_dim,drop_prob=0.1):
        super(FeedForwardLayer,self).__init__()
        self.layers=nn.ModuleList(
            [nn.Linear(d_model,hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(hidden_dim,d_model)]
        )
    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return x

def build_ffn(args):
    d_model=args.d_model
    hidden_dim=args.ffn_hide_dim
    drop_prob=args.drop_prob

    return FeedForwardLayer(d_model,hidden_dim,drop_prob)