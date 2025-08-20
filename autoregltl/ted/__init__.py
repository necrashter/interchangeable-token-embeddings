import torch

from . model import TransformerConfig, Transformer
from autoregltl.embedding import EmbedderConfig
from . train import TedTrainer


def create_model(args, vocab):
    config = TransformerConfig(
        vocab = vocab,
        d_embed_enc = args.d_embed_enc,
        d_embed_dec = args.d_embed_dec,
        d_ff = args.d_ff,
        ff_activation = args.ff_activation,
        dropout = args.dropout,
        num_heads = args.num_heads,
        num_layers = args.num_layers,
        layer_norm_eps = args.layer_norm_eps,
        merged_embedder = EmbedderConfig.from_args(args) if args.merged_vocab else None,
        enc_pe = args.enc_pe,
        dec_pe = args.dec_pe,
        no_pe_cross_keys = args.no_pe_cross_keys,
        # Not configurable for now
        # max_encode_length = args.max_encode_length,
        # max_decode_length = args.max_decode_length,
    )
    return Transformer(config).to(torch.device(args.device))


def load_model(path, device, **kwargs):
    return Transformer.load_pretrained(path, device=device)

def get_gen_args(args):
    return dict(
        alpha=args.alpha,
        beam_size=args.beam_size,
        gen_batch_size=args.gen_batch_size,
    )