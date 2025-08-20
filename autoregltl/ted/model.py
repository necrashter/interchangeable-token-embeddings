"""
Transformer encoder-decoder implementation
Based on https://github.com/tensorflow/models/tree/master/official/nlp/transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from autoregltl.ltl.vocab import EncDecVocab, CharVocab, MergedLTLVocab
from autoregltl.dataset import EncDecLTLCollator, EncDecLTLDataset
from autoregltl.embedding import EmbedderConfig, DynamicEmbedder
from autoregltl import positional_encoding as pe
from autoregltl.ltl.parser import ParseError, ltl_formula, ltl_trace
from .layers import attention
from .beam_search import BeamSearch

import json
import os
from tqdm.auto import tqdm
from dataclasses import dataclass, field, asdict
from collections import namedtuple
from typing import Dict, Union, Any, Optional, Tuple, List


@dataclass
class TransformerConfig:
    vocab: EncDecVocab | MergedLTLVocab

    d_embed_enc: int  # dimension of encoder embedding
    d_embed_dec: int  # dimension of decoder embedding
    d_ff: int  # hidden dimension of feed-forward networks
    ff_activation: str  # activation function used in feed-forward networks
    dropout: float  # percentage of droped out units
    num_heads: int  # number of attention heads
    num_layers: int  # number of encoder / decoder layers
    layer_norm_eps: float

    merged_embedder: Optional[EmbedderConfig] = None

    # Used for constructing the positional encoding buffers
    max_encode_length: int = 1024  # maximum length of input sequence
    max_decode_length: int = 1024  # maximum length of target sequence

    tree_pos_enc: bool = False

    datatype: str = 'float32'  # datatype for floating point computations

    enc_pe: str = 'sinusoid'  # type of positional encoding for encoder
    dec_pe: str = 'sinusoid'  # type of positional encoding for decoder
    no_pe_cross_keys: bool = False  # whether to use positional encoding for cross-attention keys

    # The methods needed by Trainer
    def to_json_string(self):
        return json.dumps(asdict(self))
    def to_dict(self):
        return asdict(self)
    
    def __post_init__(self):
        self.dtype = getattr(torch, self.datatype)
        # For loading from dictionary
        if isinstance(self.merged_embedder, dict):
            self.merged_embedder = EmbedderConfig(**self.merged_embedder)

        if isinstance(self.vocab, dict):
            if self.merged_embedder:
                self.vocab = MergedLTLVocab(**self.vocab)
            else:
                self.vocab = EncDecVocab(CharVocab(**self.vocab["inp"]), CharVocab(**self.vocab["out"]))

        if self.d_embed_dec is None:
            self.d_embed_dec = self.d_embed_enc
        self.d_embed_enc -= self.d_embed_enc % self.num_heads  # round down
        self.d_embed_dec -= self.d_embed_dec % self.num_heads  # round down


def get_activation(activation):
    """
    Args:
        activation: str, name of the activation function
    """
    if activation =='relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f'Unknown activation function {activation}')


def create_padding_mask(input, pad_id, dtype=torch.float32):
    """
    Args:
        input: int tensor with shape (batch_size, input_length)
        pad_id: int, encodes the padding token
        dtype: data type of look ahead mask
    Returns:
        padding mask with shape (batch_size, 1, 1, input_length) that indicates padding with 1 and 0 everywhere else
    """
    mask = (input == pad_id).to(dtype)
    return mask.unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size, device, dtype=torch.float32):
    """
    creates a look ahead mask that masks future positions in a sequence, e.g., [[[[0, 1, 1], [0, 0, 1], [0, 0, 0]]]] for size 3
    Args:
        size: int, specifies the size of the look ahead mask
        device: torch.device, device where the tensors reside
        dtype: data type of look ahead mask
    Returns:
        look ahead mask with shape (1, 1, size, size) that indicates masking with 1 and 0 everywhere else
    """
    mask = torch.triu(torch.ones(size, size, device=device, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(1)


class TransformerEncoderLayer(nn.Module):
    """A single encoder layer of the Transformer that consists of two sub-layers: a multi-head
    self-attention mechanism followed by a fully-connected feed-forward network. Both sub-layers
    employ a residual connection followed by a layer normalization."""

    def __init__(self, config: TransformerConfig):
        """
        Args:
            config: hyperparameter dictionary containing the following keys:
                d_embed_enc: int, dimension of encoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
        """
        super(TransformerEncoderLayer, self).__init__()
        self.multi_head_attn = attention.MultiHeadAttention(config.d_embed_enc, config.num_heads, config.enc_pe)

        self.ff = nn.Sequential(
            nn.Linear(config.d_embed_enc, config.d_ff),
            get_activation(config.ff_activation),
            nn.Linear(config.d_ff, config.d_embed_enc)
        )

        self.norm_attn = nn.LayerNorm(config.d_embed_enc, eps=config.layer_norm_eps)
        self.norm_ff = nn.LayerNorm(config.d_embed_enc, eps=config.layer_norm_eps)

        self.dropout_attn = nn.Dropout(config.dropout)
        self.dropout_ff = nn.Dropout(config.dropout)

    def forward(self, input, mask):
        """
        Args:
            input: float tensor with shape (batch_size, input_length, d_embed_dec)
            mask: float tensor with shape (batch_size, 1, 1, input_length)
        """
        attn, attn_weights = self.multi_head_attn(input, input, input, mask)
        attn = self.dropout_attn(attn)
        norm_attn = self.norm_attn(attn + input)

        ff_out = self.ff(norm_attn)
        ff_out = self.dropout_ff(ff_out)
        norm_ff_out = self.norm_ff(ff_out + norm_attn)

        return norm_ff_out, attn_weights


class TransformerDecoderLayer(nn.Module):
    """A single decoder layer of the Transformer that consists of three sub-layers: a multi-head
    self-attention mechanism followed by a multi-head encoder-decoder-attention mechanism followed
    by a fully-connected feed-forward network. All three sub-layers employ a residual connection
    followed by a layer normalization."""

    def __init__(self, config):
        """
        Args:
            config: hyperparameter dictionary containing the following keys:
                d_embed_dec: int, dimension of decoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
        """
        super(TransformerDecoderLayer, self).__init__()
        self.multi_head_self_attn = attention.MultiHeadAttention(config.d_embed_dec, config.num_heads, config.dec_pe)
        self.multi_head_enc_dec_attn = attention.MultiHeadAttention(config.d_embed_dec, config.num_heads, config.dec_pe)
        self.no_pe_cross_keys = config.no_pe_cross_keys

        self.ff = nn.Sequential(
            nn.Linear(config.d_embed_dec, config.d_ff),
            get_activation(config.ff_activation),
            nn.Linear(config.d_ff, config.d_embed_dec)
        )

        self.norm_self_attn = nn.LayerNorm(config.d_embed_dec, eps=config.layer_norm_eps)
        self.norm_enc_dec_attn = nn.LayerNorm(config.d_embed_dec, eps=config.layer_norm_eps)
        self.norm_ff = nn.LayerNorm(config.d_embed_dec, eps=config.layer_norm_eps)

        self.dropout_self_attn = nn.Dropout(config.dropout)
        self.dropout_enc_dec_attn = nn.Dropout(config.dropout)
        self.dropout_ff = nn.Dropout(config.dropout)

    def forward(self, input, enc_output, look_ahead_mask, padding_mask, cache=None):
        """
        Args:
            input: float tensor with shape (batch_size, target_length, d_embed_dec)
            enc_output: float tensor with shape (batch_size, input_length, d_embed_enc)
            look_ahead_mask: float tensor with shape (1, 1, target_length, target_length)
            padding_mask: float tensor with shape (batch_size, 1, 1, input_length)
            cache: dict
        """
        # Note that cache is transposed compared to attention: (batch_size, seq_len, num_heads, d_heads)
        past_queries = cache['keys'].size(1) if cache is not None else 0

        self_attn, self_attn_weights = self.multi_head_self_attn(
            input, input, input,
            look_ahead_mask,
            cache,
            past_queries = past_queries,
        )
        self_attn = self.dropout_self_attn(self_attn)
        norm_self_attn = self.norm_self_attn(self_attn + input)

        enc_dec_attn, enc_dec_attn_weights = self.multi_head_enc_dec_attn(
            norm_self_attn, enc_output, enc_output,
            padding_mask,
            past_queries = past_queries,
            no_pe_keys = self.no_pe_cross_keys,
        )
        enc_dec_attn = self.dropout_enc_dec_attn(enc_dec_attn)
        norm_enc_dec_attn = self.norm_enc_dec_attn(
            enc_dec_attn + norm_self_attn)

        ff_out = self.ff(norm_enc_dec_attn)
        ff_out = self.dropout_ff(ff_out)
        norm_ff_out = self.norm_ff(ff_out + norm_enc_dec_attn)

        return norm_ff_out, self_attn_weights, enc_dec_attn_weights


class TransformerEncoder(nn.Module):
    """The encoder of the Transformer that is composed of num_layers identical layers."""

    def __init__(self, config: TransformerConfig):
        """
        Args:
            config: hyperparameter dictionary containing the following keys:
                d_embed_enc: int, dimension of encoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
                num_layers: int, number of encoder / decoder layers
        """
        super(TransformerEncoder, self).__init__()
        self.config = config
        self.enc_layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_layers)])

    def forward(self, x, padding_mask):
        attn_weights = {}
        for i, layer in enumerate(self.enc_layers):
            x, self_attn_weights = layer(x, padding_mask)
            attn_weights[f'layer_{i+1}'] = {}
            attn_weights[f'layer_{i+1}']['self_attn'] = self_attn_weights
        return x, attn_weights


class TransformerDecoder(nn.Module):
    """The decoder of the Transformer that is composed of num_layers identical layers."""

    def __init__(self, config: TransformerConfig):
        """
        Args:
            config: hyperparameter dictionary containing the following keys:
                d_embed_dec: int, dimension of decoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                dropout: float, percentage of droped out units
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
                num_layers: int, number of encoder / decoder layers
        """
        super(TransformerDecoder, self).__init__()
        self.dec_layers = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(config.num_layers)])

    def forward(self, x, enc_output, look_ahead_mask, padding_mask, cache=None):
        attn_weights = {}
        for i, layer in enumerate(self.dec_layers):
            layer_cache = cache[f'layer_{i}'] if cache is not None else None
            x, self_attn_weights, enc_dec_attn_weights = layer(x, enc_output, look_ahead_mask, padding_mask, layer_cache)
            attn_weights[f'layer_{i+1}'] = {}
            attn_weights[f'layer_{i+1}']['self_attn'] = self_attn_weights
            attn_weights[f'layer_{i+1}']['enc_dec_attn'] = enc_dec_attn_weights
        return x, attn_weights


class Transformer(nn.Module):
    """The Transformer that consists of an encoder and a decoder. The encoder maps the input
    sequence to a sequence of continuous representations. The decoder then generates an output
    sequence in an auto - regressive way."""

    def __init__(self, config, dtype=torch.float32):
        """
        Args:
            config: hyperparameter dictionary containing the following keys:
                d_embed_enc: int, dimension of encoder embedding
                d_embed_dec: int, dimension of decoder embedding
                d_ff: int, hidden dimension of feed-forward networks
                ff_activation: string, activation function used in feed-forward networks
                num_heads: int, number of attention heads
                num_layers: int, number of encoder / decoder layer
                max_encode_length: int, maximum length of input sequence
                max_decode_length: int, maximum lenght of target sequence
                dropout: float, percentage of droped out units
            dtype: datatype for floating point computations
        """
        super(Transformer, self).__init__()
        self.config = config

        if config.merged_embedder is not None:
            if config.d_embed_enc != config.d_embed_dec:
                raise ValueError("Cannot merge vocabularies: embedding dimensions don't match")
            merged_embedder = config.merged_embedder.build(config.d_embed_enc, config.vocab, dtype=dtype)
            embedding_func = lambda x: merged_embedder.embed(x)
            self.encoder_embedding = embedding_func
            self.decoder_embedding = embedding_func
            self.final_projection = lambda x: merged_embedder.project(x)
            self.merged_embedder = merged_embedder
            self.start_id = config.vocab.start_id
            self.pad_id = config.vocab.pad_id
        else:
            self.merged_embedder = None
            self._encoder_embedding = nn.Embedding(config.vocab.inp.size(), config.d_embed_enc)
            self.encoder_embedding = lambda x: self._encoder_embedding(x) * torch.sqrt(torch.tensor(self.config.d_embed_enc, dtype=self.dtype))
            self._decoder_embedding = nn.Embedding(config.vocab.out.size(), config.d_embed_dec)
            self.decoder_embedding = lambda x: self._decoder_embedding(x) * torch.sqrt(torch.tensor(self.config.d_embed_dec, dtype=self.dtype))
            self.final_projection = nn.Linear(config.d_embed_dec, config.vocab.out.size())
            self.start_id = config.vocab.out.start_id
            self.pad_id = config.vocab.inp.pad_id  # Input and output vocabs always have the same pad id

        self.register_buffer(
            'encoder_positional_encoding',
            pe.positional_encoding(config.max_encode_length, config.d_embed_enc),
            persistent=False,
        )
        self.encoder_dropout = nn.Dropout(config.dropout)

        self.encoder_stack = TransformerEncoder(config)

        self.register_buffer(
            'decoder_positional_encoding',
            pe.positional_encoding(config.max_decode_length, config.d_embed_dec),
            persistent=False,
        )
        self.decoder_dropout = nn.Dropout(config.dropout)

        self.decoder_stack = TransformerDecoder(config)

        self.softmax = nn.Softmax(dim=-1)

        self.dtype = dtype

    def encode(self, inputs, padding_mask, positional_encoding):
        """
        Args:
            inputs: int tensor with shape (batch_size, input_length)
            padding_mask: float tensor with shape (batch_size, 1, 1, input_length)
            positional_encoding: float tensor with shape (batch_size, input_length, d_embed_enc)
        """
        input_embedding = self.encoder_embedding(inputs)
        if positional_encoding is not None:
            input_embedding += positional_encoding
        input_embedding = self.encoder_dropout(input_embedding)
        encoder_output, attn_weights = self.encoder_stack(input_embedding, padding_mask)
        return encoder_output, attn_weights

    def decode(self, target, encoder_output, input_padding_mask):
        """
        Args:
            target: int tensor with shape (batch_size, target_length)
            encoder_output: float tensor with shape (batch_size, input_length, d_embedding)
            input_padding_mask: float tensor with shape (batch_size, 1, 1, input_length)
        
        Returns:
            logits: float tensor with shape (batch_size, target_length, out_vocab_size)
            attn_weights: dictionary with keys 'layer_i' where i is the layer number and values are float tensors with shape (batch_size, num_heads, target_length, input_length)
        """
        target_length = target.size(1)
        look_ahead_mask = create_look_ahead_mask(target_length, target.device, self.dtype)
        target_padding_mask = create_padding_mask(target, self.pad_id, self.dtype)
        look_ahead_mask = torch.maximum(look_ahead_mask, target_padding_mask)

        # shift targets to the right, insert start_id at first postion, and remove last element
        target = F.pad(target, (1, 0), value=self.start_id)[:, :-1]

        target_embedding = self.decoder_embedding(target)  # (batch_size, target_length, d_embedding)
        if self.config.dec_pe == 'sinusoid':
            target_embedding += self.decoder_positional_encoding[:, :target_length, :]
        decoder_embedding = self.decoder_dropout(target_embedding)

        decoder_output, attn_weights = self.decoder_stack(
            decoder_embedding, encoder_output, look_ahead_mask, input_padding_mask)
        output = self.final_projection(decoder_output)

        return output, attn_weights

    def forward(self, input, target, positional_encoding=None):
        """
        Args:
            input: int tensor with shape (batch_size, input_length)
            (optional) target: int tensor with shape (batch_size, target_length)
            padding mask with shape (batch_size, 1, 1, input_length) that indicates padding with 1 and 0 everywhere else
            (optional) positional_encoding: float tensor with shape (batch_size, input_length, d_embed_enc), custom postional encoding
        """
        if self.training and self.merged_embedder:
            self.merged_embedder.prepare()

        input_padding_mask = create_padding_mask(input, self.pad_id, self.dtype)

        if positional_encoding is None and self.config.enc_pe == 'sinusoid':
            assert not self.config.tree_pos_enc
            seq_len = input.size(1)
            positional_encoding = self.encoder_positional_encoding[:, :seq_len, :]
        encoder_output, encoder_attn_weights = self.encode(input, input_padding_mask, positional_encoding)

        logits, _ = self.decode(target, encoder_output, input_padding_mask)
        return logits

    def generate(
            self,
            input,
            max_decode_length,
            positional_encoding=None,
            alpha=1.0,
            beam_size=1,
            syntax_enforcer=None,
        ):
        """
        Args:
            input_padding_mask: flaot tensor with shape (batch_size, 1, 1, input_length)
            alpha: float, strength of normalization in beam search algorithm
            beam_size: int, number of beams kept by beam search algorithm
        """
        batch_size = input.size(0)

        input_padding_mask = create_padding_mask(input, self.pad_id, self.dtype)

        if positional_encoding is None and self.config.enc_pe == 'sinusoid':
            seq_len = input.size(1)
            positional_encoding = self.encoder_positional_encoding[:, :seq_len, :]
        encoder_output, encoder_attn_weights = self.encode(input, input_padding_mask, positional_encoding)

        def logits_fn(ids, i, cache):
            """
            Args:
                ids: int tensor with shape (batch_size * beam_size, index + 1)
                index: int, current index
                cache: dictionary storing encoder output, previous decoder attention values
            Returns:
                logits with shape (batch_size * beam_size, vocab_size) and updated cache
            """
            # set input to last generated id
            decoder_input = ids[:, -1:]
            decoder_input = self.decoder_embedding(decoder_input)
            if self.config.dec_pe == 'sinusoid':
                decoder_input += self.decoder_positional_encoding[:, i:i + 1, :]

            look_ahead_mask = create_look_ahead_mask(max_decode_length, ids.device, self.dtype)
            self_attention_mask = look_ahead_mask[:, :, i:i + 1, :i + 1]
            decoder_output, _ = self.decoder_stack(
                decoder_input, cache['encoder_output'], self_attention_mask, cache['input_padding_mask'], cache)
            output = self.final_projection(decoder_output)
            output = output.squeeze(1)
            return output, cache

        initial_ids = torch.ones(batch_size, dtype=torch.int32, device=encoder_output.device) * self.start_id

        num_heads = self.config.num_heads
        d_heads = self.config.d_embed_dec // num_heads
        # create cache structure for decoder attention
        cache = {
            f'layer_{layer}': {
                'keys': torch.zeros(batch_size, 0, num_heads, d_heads, device=encoder_output.device, dtype=self.dtype),
                'values': torch.zeros(batch_size, 0, num_heads, d_heads, device=encoder_output.device, dtype=self.dtype)
            } for layer in range(self.config.num_layers)
        }
        # add encoder output to cache
        cache['encoder_output'] = encoder_output
        cache['input_padding_mask'] = input_padding_mask

        beam_search = BeamSearch(
            logits_fn,
            batch_size,
            encoder_output.device,
            syntax_enforcer,
            max_decode_length,
            self.start_id,
            self.config.vocab.eos_id if self.merged_embedder else self.config.vocab.out.eos_id,
            self.merged_embedder.output_vocab_size if self.merged_embedder else self.config.vocab.out.size(),
            alpha,
            beam_size,
            self.dtype,
        )
        decoded_ids, scores = beam_search.search(initial_ids, cache)

        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        # compute attention weights
        _, decoder_attn_weights = self.decode(top_decoded_ids, encoder_output, input_padding_mask)

        return {'outputs': top_decoded_ids, 'scores': top_scores, 'enc_attn_weights': encoder_attn_weights, 'dec_attn_weights': decoder_attn_weights}
    
    @torch.inference_mode()
    def generate_predictions(self, dataset, max_length, gen_args, leave_tqdm=True, prepare_embedder=True):
        self.eval()
        if prepare_embedder:
            enc_dec_dataset = EncDecLTLDataset(
                filename=None,
                vocab=self.config.vocab,
                max_formula_length=None,
                max_trace_length=None,
                tree_pos_enc=self.config.tree_pos_enc,
                pairs=dataset.data[:len(dataset.data)//10],
            )
            self.set_median_w(enc_dec_dataset)

        for param in self.parameters():
            model_device = param.device
            break

        vocab = self.config.vocab
        if isinstance(vocab, EncDecVocab):
            input_encode = lambda x: vocab.inp.encode(x, prepend_start_token=False)
            output_decode = lambda x: vocab.out.decode(x)
        elif isinstance(vocab, MergedLTLVocab):
            input_encode = lambda x: vocab.encode_ltl(x, eos=True)
            output_decode = lambda x: vocab.decode(x)
        else:
            raise ValueError(f"Unsupported vocab type: {type(vocab)}")

        predictions = []
        if "gen_batch_size" in gen_args:
            gen_args = gen_args.copy()
            batch_size = gen_args.pop("gen_batch_size")
        else:
            batch_size = 512
        dataloader = DataLoader(dataset, batch_size=batch_size)
        with tqdm(total=len(dataset), desc="Predict", leave=leave_tqdm) as pbar:
            for (traces, formulas) in dataloader:
                # Pad by adding pad tokens to the right (end)
                input_ids = [torch.tensor(input_encode(formula), dtype=torch.long) for formula in formulas]
                input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
                positional_encoding = None
                if self.config.tree_pos_enc:
                    positional_encoding = []
                    max_seq_len = input_ids.size(1)
                    for formula in formulas:
                        position_list = ltl_formula(formula, 'network-polish').binary_position_list(format='lbt', add_first=True)
                        padded_position_list = [l + [0] * (self.config.d_embed_enc - len(l)) for l in position_list]
                        pe = torch.tensor(padded_position_list, dtype=torch.float32)
                        pe = F.pad(pe, (0, self.config.d_embed_enc - pe.size(-1), 0, max_seq_len - pe.size(-2)))
                        positional_encoding.append(pe)
                    positional_encoding = torch.stack(positional_encoding, dim=0)
                out = self.generate(
                    input=input_ids.to(model_device),
                    # +1 for start token
                    max_decode_length = max_length + 1,
                    positional_encoding = positional_encoding.to(model_device) if positional_encoding is not None else None,
                    **gen_args,
                )['outputs']
                for prediction, trace, formula in zip(out.tolist(), traces, formulas):
                    prediction = output_decode(prediction)
                    # formula trace target
                    predictions.append((prediction, trace, formula))
                pbar.update(len(formulas))
        return predictions

    @classmethod
    def load_pretrained(cls, directory, dtype=torch.float32, device=None, **kwargs):
        if not os.path.exists(directory):
            raise FileNotFoundError("Model directory is not found")

        with open(os.path.join(directory, 'config.json'), 'r') as f:
            config_data = json.load(f)
        config = TransformerConfig(**config_data)

        model = cls(config, dtype=dtype, **kwargs)
        if device is not None:
            model = model.to(device)
        state_dict = torch.load(os.path.join(directory, "pytorch_model.bin"), map_location=device)
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=4)
    
    @torch.inference_mode()
    def set_median_w(self, dataset, batch_size=512, repeat_count=10):
        """
        1. Evaluate cross entropy loss on the given dataset multiple times
        2. Set the merged_embedder.w to the median of the loss
        """
        assert getattr(self, 'merged_embedder', None) is not None

        self.eval()
        for param in self.parameters():
            device = param.device
            break

        if self.config.tree_pos_enc:
            data_collator = EncDecLTLCollator(self.config.d_embed_enc)
        else:
            data_collator = EncDecLTLCollator()
        crossent = torch.nn.CrossEntropyLoss(reduction='sum')

        evals = []
        for repetition in tqdm(range(repeat_count), desc="Reps"):
            self.merged_embedder.prepare()
            w_matrix = self.merged_embedder.w.detach()

            # Compute cross entropy loss on all_dataset
            # Initalize dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=data_collator,
            )
            loss = 0
            for inputs in dataloader:
                if self.config.tree_pos_enc:
                    logits = self(inputs["input_ids"].to(device), inputs["target_ids"].to(device), inputs["pe"].to(device))
                else:
                    logits = self(inputs["input_ids"].to(device), inputs["target_ids"].to(device))
                labels = torch.where(inputs["target_ids"] == self.pad_id, -100, inputs["target_ids"]).to(device)
                loss += crossent(logits.view(-1, logits.size(-1)), labels.view(-1)).item() / len(dataset)

            evals.append({
                "w_matrix": w_matrix,
                "loss": loss,
            })

        evals = sorted(evals, key=lambda x: x["loss"])
        median = len(evals) // 2
        # Get w_matrix of median
        w_matrix = evals[median]["w_matrix"]
        # Set w_matrix of model to median
        self.merged_embedder.w.set_(w_matrix)
        return evals[median]