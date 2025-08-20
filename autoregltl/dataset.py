import os
import re
import sys
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Union, Any, Optional, Tuple, List
from tqdm.auto import tqdm
from torch.utils.data import Dataset

import urllib.request

from autoregltl.ltl.vocab import MergedLTLVocab, EncDecVocab
from autoregltl.ltl.chars import CHARS


def download_dataset(dataset_name, split, dataset_dir):
    url_lookup = {
        'na-5-ts-35-nf-1m-lbt-sat': 'https://storage.googleapis.com/deepltl_data/data/ltl_traces/na-5-ts-35-nf-1m-lbt-sat/',
        'na-5-ts-35-50-nf-20k-lbt-sat': 'https://storage.googleapis.com/deepltl_data/data/ltl_traces/na-5-ts-35-50-nf-20k-lbt-sat/',
    }

    # Check if split already exists
    split_file = os.path.join(dataset_dir, split + '.txt')
    if not os.path.isfile(split_file):
        if dataset_name not in url_lookup:
            print("Cannot download this dataset:", dataset_name)
            return
        os.makedirs(dataset_dir, exist_ok=True)
        print(f'Downloading dataset {dataset_name}/{split}')
        urllib.request.urlretrieve(url_lookup[dataset_name] + split + '.txt', split_file)


class SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def read_pairs(
    filename,
    max_formula_length,
    max_trace_length,
    max_samples = None,
    print_filtered = True,
):
    """
    Expects data file to have formula\ntrace\n format
    """
    filtered = 0
    pairs = []
    with open(filename, 'r') as file:  # expect formula\ntrace\n format
        for formula_line in file:
            if formula_line == '\n':
                break
            formula_line = formula_line.strip()
            if max_formula_length >= 0 and len(formula_line) > max_formula_length:
                filtered += 1
                continue
            pairs.append(formula_line)

    if max_samples is not None:
        pairs = pairs[:max_samples]

    if print_filtered: print("Filtered out", filtered, "samples")
    return pairs


class RawLTLDataset(SeqDataset):
    """Dataset that consists of pairs of a LTL formula and a satisfying trace."""
    def __init__(
        self, 
        filename,
        max_formula_length,
        max_trace_length,
        max_samples = None,
        print_filtered = True,
    ):
        pairs = read_pairs(filename, max_formula_length, max_trace_length, max_samples, print_filtered)
        super().__init__(pairs)


class DecoderLTLDataset(SeqDataset):
    def __init__(
        self, 
        filename,
        vocab: MergedLTLVocab,
        max_formula_length,
        max_trace_length,
        max_samples = None,
    ):
        pairs = read_pairs(filename, max_formula_length, max_trace_length, max_samples)

        def process_pair(trace_str):
            trace = vocab.encode_trace(trace_str + ";")
            formula = vocab.encode_ltl(trace_str)
            # No need to feed EOS in input
            input_ids = trace + formula
            labels = ([-100] * (len(trace)-1)) + formula + [vocab.eos_id]
            return torch.tensor(input_ids), torch.tensor(labels)

        data = [process_pair(pair) for pair in tqdm(pairs, desc=os.path.basename(filename))]
        super().__init__(data)


@dataclass
class DecoderLTLCollator:
    pad_token_id: int = 0
    ignore_label: int = -100

    def __call__(self, instances):
        input_ids, labels = zip(*instances)  # Unzip
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.ignore_label)
        return {"input_ids": input_ids, "labels": labels}


class EncDecLTLDataset(SeqDataset):
    def __init__(
        self, 
        filename,
        vocab: EncDecVocab,
        max_formula_length,
        max_trace_length,
        max_samples = None,
        pairs = None,
    ):
        if pairs is None:
            pairs = read_pairs(filename, max_formula_length, max_trace_length, max_samples)

        if isinstance(vocab, EncDecVocab):
            def process_pair(formula_str):
                # Input is trace, output is ltl
                formula = vocab.out.encode(formula_str, prepend_start_token=False)
                return torch.tensor(formula), torch.tensor(formula)
        elif isinstance(vocab, MergedLTLVocab):
            def process_pair(formula_str):
                formula = vocab.encode_ltl(formula_str, eos=True)
                return torch.tensor(formula), torch.tensor(formula)
        else:
            raise ValueError(f"Unsupported vocab type: {type(vocab)}")

        if filename is not None:
            iterator = tqdm(pairs, desc=os.path.basename(filename))
        else:
            iterator = pairs
        data = [process_pair(pair) for pair in iterator]
        super().__init__(data)


@dataclass
class EncDecLTLCollator:
    """
    A collate function that pads the input sequences to the longest sequence in the batch.
    """
    d_embed_enc: Optional[int] = None

    def __call__(self, batch):
        """
        Args:
            batch: list of (input, target) or (input, target, positional_encoding)

        Returns:
            A dictionary:
                input_ids: int tensor with shape (batch_size, input_length)
                target_ids: int tensor with shape (batch_size, target_length)
                pe: float tensor with shape (batch_size, input_length, d_embed_enc), custom postional encoding
        """
        # Pad by adding zeros to the end
        max_input_length = max(map(lambda x: x[0].size(0), batch))
        xs = torch.stack([F.pad(x[0], (0, max_input_length - x[0].size(0)), "constant", 0) for x in batch], dim=0)
        max_target_length = max(map(lambda x: x[1].size(0), batch))
        ys = torch.stack([F.pad(x[1], (0, max_target_length - x[1].size(0)), "constant", 0) for x in batch], dim=0)
        pe = None
        if self.d_embed_enc is not None:
            pe = torch.stack([F.pad(x[2], (0, self.d_embed_enc - x[2].size(-1), 0, max_input_length - x[2].size(-2)), "constant", 0) for x in batch], dim=0)
        return {
            "input_ids": xs,
            "target_ids": ys,
            "pe": pe,
        }


def get_dataset_vocab(args, config=None):
    if args.ds_name is None:
        raise ValueError('No dataset is specified')

    aps = ['a', 'b', 'c', 'd', 'e']
    if args.ds_name == 'prop-60-no-derived':
        aps += ['f', 'g', 'h', 'i', 'j']
    
    # Match the 6 in ltlx-6ap-etc
    # \b is word boundary
    if matches := re.findall(r'\b(\d+)ap\b', args.ds_name):
        ap_count = int(matches[0])
        aps = CHARS[:ap_count]

    if getattr(args, 'merged_vocab', False) or getattr(config, 'merged_embedder', None) is not None:
        merge_tokens = args.merge_tokens if config is None else config.vocab.merge_tokens
        dynamic_aps = args.dynamic_aps if config is None else config.vocab.dynamic_aps
        kwargs = {} if args.decoder_only else {"use_start_token": True, "use_pad_token": True}
        vocab = MergedLTLVocab(aps=aps, merge_tokens=merge_tokens, dynamic_aps=dynamic_aps, **kwargs)
        print(f"[get_dataset_vocab] Vocab size: {vocab.size()} (merged tokens: {merge_tokens})")
        return vocab
    else:
        # encoder-decoder
        vocab = EncDecVocab.create_ltl_vocab(aps=aps)
        print(f"[get_dataset_vocab] Input vocab size: {vocab.inp.size()}, output vocab size: {vocab.out.size()}")
        return vocab


def get_dataset(args, split, dataset_class, **kwargs):
    """
    Returns:
        the datasets corresponding to the given split according to args
    """
    # Dataset specification
    if args.ds_name is None:
        sys.exit('No dataset specified\n')
    else:
        if args.ds_name == 'ltl-35' or args.ds_name == 'prop-35':
            max_formula_length = 35
            dataset_name = 'na-5-ts-35-nf-1m-lbt-sat'
        elif args.ds_name == 'ltl-50-test' or args.ds_name == 'prop-50-test':
            max_formula_length = 50
            if split != 'test':
                sys.exit(f'Dataset {args.ds_name} can only be used in test mode\n')
            dataset_name = 'na-5-ts-35-50-nf-20k-lbt-sat'
        elif os.path.exists(os.path.join(args.data_dir, args.ds_name)):
            max_formula_length = -1
            dataset_name = args.ds_name
        else:
            sys.exit(f'{args.ds_name} is not a valid dataset\n')

    # data_dir = data_dir if data_dir is not None else os.path.join(os.path.dirname(__file__), '../../../data')
    dataset_dir = os.path.join(args.data_dir, dataset_name)

    download_dataset(dataset_name, split, dataset_dir)

    dataset_args = {
        'max_formula_length': max_formula_length,
        'max_trace_length': args.max_trace_length,
    }
    dataset_args.update(kwargs)

    return dataset_class(os.path.join(dataset_dir, split + '.txt'), **dataset_args)