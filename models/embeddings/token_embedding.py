from torch import nn



class TokenEmbedding(nn.Embedding):
    """
    nn.Embedding模块,专门用于embed
    """
    def __init__(self,vocab_len:int, d_model:int, pad_idx):
        """
        vacab_len: 词表长度
        d_model: 转化后一个词对应的向量长度
        padding_idx: 表示padd的数字
        在这种情况下，padding_idx=0 表示在使用这个嵌入层时，索引为 0 的位置将被视为填充位置。在将序列传递给嵌入层时，任何包含索引为 0 的位置都将被填充向量代替，这有助于保持序列的统一长度。
        """
        super(TokenEmbedding,self).__init__(vocab_len,d_model,padding_idx=pad_idx)
