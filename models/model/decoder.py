import torch 
import torch.nn as nn

from ..blocks import *
from ..layers import *
class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, ffn_layer,norm_layer1,dropout1,norm_layer2,dropout2,norm_layer3,dropout3) -> None:
        super(DecoderLayer,self).__init__()
        self.self_attention=self_attention
        self.cross_attention=cross_attention
        self.ffn_layer=ffn_layer

        self.norm_layer1=norm_layer1
        self.dropout1=dropout1
        self.norm_layer2=norm_layer2
        self.dropout2=dropout2
        self.norm_layer3=norm_layer3
        self.dropout3=dropout3
       
    def forward(self,tgt,src,tgt_mask=None,tgt_src_mask=None):
        tgt_encode,_=self.self_attention(tgt,tgt_mask)
        tgt_encode=self.dropout1(tgt_encode)
        tgt=self.norm_layer1(tgt+tgt_encode)

        res,_=self.cross_attention(q=tgt, k=src, v=src,mask=tgt_src_mask)
        res=self.dropout2(res)
        res=self.norm_layer2(tgt+res)

        _res=res
        res=self.ffn_layer(res)
        res=self.dropout3(res)
        res=self.norm_layer3(res+_res)

        return res

class Decoder(nn.Module):
    def __init__(self,num_layers,decoderLayer):
        super(Decoder,self).__init__()
        self.Layers=nn.ModuleList([decoderLayer for _ in range(num_layers)])

    def forward(self, tgt, src, tgt_mask, tgt_src_mask):
        for layer in self.Layers:
            tgt=layer(tgt,src,tgt_mask,tgt_src_mask)
        
        return tgt

def build_decoder(args):
    d_model=args.d_model
    drop_prob=args.drop_prob
    self_attention=build_attention(args)
    cross_attention=build_attention(args)
    ffn_layer=build_ffn(args)
    norm_drops=[]
    for _ in range(3):
        norm_drops.extend([nn.LayerNorm(d_model,eps=1e-5),nn.Dropout(drop_prob)])
    decoder_layer=DecoderLayer(self_attention,cross_attention,ffn_layer,*norm_drops)
    decoder=Decoder(args.num_decoder_layers, decoder_layer)
    return decoder
