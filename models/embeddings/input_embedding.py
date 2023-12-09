import torch.nn as nn
from .position_encoding import *
from .token_embedding import *







class InputEmbedding(nn.Module):
    def __init__(self,token_embedding, position_encoding,drop_prob=0.1):
        super(InputEmbedding,self).__init__()

        self.token_embedding=token_embedding
        self.position_encoding=position_encoding

        self.drop_out=nn.Dropout(p=drop_prob)


    def forward(self,x):
        x_embed=self.token_embedding(x) #[bs, len] -> [bs,len,d_model]
        pos_code=self.position_encoding(x) # [len,d_model]
        # 会自动在每个维度上分别相加
        return x_embed+pos_code


def build_inputEmbedding(vocab_len=None,d_model=None,pad_idx=0,device="cuda",args=None):
    vocab_len=vocab_len if vocab_len else args.vocab_len
    d_model=d_model if d_model else args.d_model
    device=device if device else args.device
    max_len=args.max_seq_len

    token_emb=TokenEmbedding(vocab_len,d_model,pad_idx)
    pos_encode=PositionEncoding(max_len,d_model, device)

    return InputEmbedding(token_emb,pos_encode,args.drop_prob)
