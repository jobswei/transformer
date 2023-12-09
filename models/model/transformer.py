import torch
import torch.nn as nn

from ..embeddings import *
from ..model import *
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
    # 建议最好在最外层的builder，也就是最大框架的builder，这里就是transformer，的传参用args，其余的都用实际的名称
    # 这样内外的界限就很明显，看起来很规整
    # 但实际上都用args也有好处，比如最内层的模块增加了一个参数，只需把这个参数加在args上，不用一层一层修改传参了
    linear=nn.Linear(args.d_model, args.tgt_vocab_size)
    src_embedding=build_inputEmbedding(mode="src", args=args)
    tgt_embedding=build_inputEmbedding(mode="tgt", args=args)
    encoder=build_encoder(args)
    decoder=build_decoder(args)
    transformer=Transformer(linear, src_embedding,tgt_embedding,encoder,decoder,src_pad_idx=args.src_pad_idx, tgt_pad_idx=args.tgt_pad_idx)
    return transformer
