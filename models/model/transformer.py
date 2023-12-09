import torch
import torch.nn as nn

from ..embeddings import *

class Transformer(nn.Module):
    def __init__(self,linear,src_embedding,tgt_embedding, encoder,decoder,src_pad_idx,tgt_pad_idx) -> None:
        super(Transformer,self).__init__()
        self.src_embedding=src_embedding
        self.tgt_embedding=tgt_embedding
        self.encoder=encoder
        self.decoder=decoder
        self.src_pad_idx=src_pad_idx
        self.tgt_pad_idx=tgt_pad_idx
        self.linear=linear
    def forward(self,src, tgt,):
        src_mask=self.get_src_mask(src)
        tgt_mask=self.get_tgt_mask(tgt)
        src=self.src_embedding(src)
        tgt=self.tgt_embedding(tgt)
        src_encode=self.encoder(src,src_mask)
        tgt_decode=self.decoder(tgt,src_encode,tgt_mask,src_mask)
        tgt=self.linear(tgt_decode)
        return tgt



    
    def get_src_mask(self,src):
        mask=(src!=self.src_pad_idx).to(torch.float32).unsqueeze(1)
        mask=mask.transpose(1,2)@mask
    def get_tgt_mask(self,tgt):
        bs,tgt_len=tgt.size()
        time_mask=torch.tril(torch.ones(tgt_len, tgt_len)).to(torch.int).to(tgt.device)
        self_mask=(tgt!=self.tgt_pad_idx).to(torch.float32).unsqueeze(1)
        self_mask=self_mask.transpose(1,2)@self_mask
        mask=time_mask*self_mask
        return mask


def build_transformer(args):
    linear=nn.Linear(args.d_model, args.tgt_vocab_size)
    src_embedding=build_inputEmbedding(args.src_vocab_size,args.d_model,args.src_pad_idx,args.device, args)
    tgt_embedding=build_inputEmbedding(args.tgt_vocab_size,args.d_model,args.tgt_pad_idx,args.device, args)
    encoder=None
    decoder=None
    transformer=Transformer(linear, src_embedding,tgt_embedding,encoder,decoder,src_pad_idx=args.src_pad_idx, tgt_pad_idx=args.tgt_pad_idx)
    return transformer
