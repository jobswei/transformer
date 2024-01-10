import argparse
from models.model import build_transformer
import torch

def get_args_parser():
    parser=argparse.ArgumentParser("Set Transformer Paras", add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=90, type=int)

    parser.add_argument('--d_model', default=256, type=int)
    parser.add_argument('--n_head', default=4, type=int)
    parser.add_argument('--ffn_hide_dim', default=512, type=int)
    parser.add_argument('--drop_prob', default=0.1, type=float)

    parser.add_argument('--num_encode_layers', default=6, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)

    parser.add_argument('--max_seq_len', default=1000, type=int)
    parser.add_argument('--src_vocab_size', default=90, type=int)
    parser.add_argument('--tgt_vocab_size', default=90, type=int)
    parser.add_argument('--src_pad_idx', default=0, type=int)
    parser.add_argument('--tgt_pad_idx', default=0, type=int)

    parser.add_argument('--device', default="cuda", type=str)
    
    parser.add_argument('--output_dir', default="./worke_dir", type=str)


    return parser

def main(args):
    model=build_transformer(args)
    model.to(args.device)
    # print(model)
    src=torch.randint(0,90,(2,10)).to(args.device)
    src=src.to(torch.int)
    print(src)
    tgt=torch.zeros(2,1).to(args.device).to(torch.int)
    res=model(src,tgt)
    print(res)


if __name__=="__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
