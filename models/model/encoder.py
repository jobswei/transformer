import torch
import torch.nn as nn

from ..blocks import *

class EncoderLayer(nn.Module):
    def __init__(self,ffn_layer,self_attention,norm_layer1,dropout1,norm_layer2,dropout2):
        super(EncoderLayer,self).__init__()
        self.attention=self_attention
        self.ffn_layer=ffn_layer
        self.norm_layer1=norm_layer1
        self.dropout1=dropout1
        self.norm_layer2=norm_layer2
        self.dropout2=dropout2

    def forward(self,x,mask=None):
        bs, seq_len, d_model=x.size()
        x_encode, score=self.attention(x,x,x,mask=mask)
        # drop要在add的前面
        x_encode=self.dropout1(x_encode)
        x=self.norm_layer1(x+x_encode)

        x_ffn=self.ffn_layer(x)
        x_ffn=self.dropout2(x_ffn)
        x=self.norm_layer2(x+x_ffn)

        return x

class Encoder(nn.Module):
    def __init__(self,num_layers,encoderLayer):
        super(Encoder,self).__init__()
        self.encoderLayer=encoderLayer
        self.num_layers=num_layers
        self.layers=nn.ModuleList([encoderLayer for _ in range(num_layers)])

    def forward(self,x,mask=None):
        for layer in self.layers:
            x=layer(x,mask)
        return x


def build_encoder(args):
    ffn_layer=None
    self_attention=bui
    encoderLayer=EncoderLayer(ffn_layer)