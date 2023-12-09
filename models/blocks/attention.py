import torch.nn as nn
import torch




class MultiHeadAttention(nn.Module):
    def __init__(self,d_model, n_head, attention):
        super(MultiHeadAttention,self).__init__()
        self.n_head=n_head
        self.d_model=d_model
        self.attention=attention

        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)

        self.w_concat=nn.Linear(d_model,d_model)

    def forward(self,q,k,v,mask=None):
        q=self.w_q(q)
        k=self.w_v(k)
        v=self.w_v(v)

        q,k,v=self.split(q),self.split(k),self.split(v)

        res, score=self.attention(q, k, v, mask)
        res,score=self.concat(res), self.concat(score)
        res=self.w_concat(res)

        return res, score
    
    def concat(self,x:torch.Tensor):
        bs, n_head, seq_len, hide_dim=x.size()
        # .contiguous(): 保证张量的储存位置在内存中是连续的
        x=x.transpose(1,2).contiguous().view(bs,seq_len,n_head*hide_dim)
        return x

    def split(self,x:torch.Tensor):
        bs, seq_len, d_model = x.size()
        hide_dim=d_model//self.n_head
        x=x.view(bs,seq_len,self.n_head,hide_dim).transpose(1,2)
        # 因为要在最后两个维度做矩阵乘法。它的意义是不同字母彼此之间的相关分数，所以一个维度是字母，一个维度是hide_dim。
        # n_head挪到前面，就是在每一个head上独立的进行attention操作
        return x


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention,self).__init__()
        self.softmax=nn.Softmax(dim=-1)
    

    def forward(self, q, k, v, mask=None):
        bs, n_head, seq_len, hide_dim = q.size()

        k=k.transpose(2,3)
        score=q@k /torch.sqrt(hide_dim)
        
        if mask is not None:
            score=score.masked_fill(mask==0,-100000) # masked_fill 方法

        score=self.softmax(score)
        res = score @ v

        return res, score

def build_attention(args):
    d_model=args.d_model
    n_head=args.n_head

    attention=ScaleDotProductAttention()
    mh_attention=MultiHeadAttention(d_model,n_head,attention)

    return mh_attention


