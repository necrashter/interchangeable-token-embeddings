import torch
import numpy as np
import os
import sys
import json
import shutil
import traceback
from glob import glob
from collections import defaultdict
from typing import Dict, Union, Any, Optional, Tuple, List
from tqdm.auto import tqdm
from pprint import pprint

from autoregltl.ltl.enforcer import LTLSyntaxEnforcerConfig
from autoregltl import dataset
from autoregltl.ltl import trace_check
from autoregltl.utils import describe_statistics, tictoc_histogram, init_plot_font
from autoregltl.eval import get_result_dir_name

from itertools import permutations

def generate_equivalent_expressions(expression, aps):
    # Extract unique variables from the expression
    unique_variables = sorted(set([ch for ch in expression if ch.isalpha()]))
    
    # Generate all permutations of valid variable names of the length of unique variables
    permuted_variables = permutations(aps, len(unique_variables))
    
    # List to hold all equivalent expressions
    equivalent_expressions = []
    
    # Replace variables with each permutation
    for perm in permuted_variables:
        translation_map = {original: new for original, new in zip(unique_variables, perm)}
        new_expression = ''.join([translation_map.get(ch, ch) for ch in expression])
        equivalent_expressions.append(new_expression)
    
    return equivalent_expressions


class ResymbolizeDataset(dataset.SeqDataset):
    def __init__(self, dataset, aps):
        self.base_dataset = dataset
        self.aps = aps
        new_data = []
        self.group_sizes = []
        for trace, formula in dataset.data:
            resyms = generate_equivalent_expressions(trace, aps)
            self.group_sizes.append(len(resyms))
            # Formula is not required for generation
            new_data += [(trace, formula) for trace in resyms]
        print("Original dataset size:", len(dataset.data))
        print("Permutations dataset size:", len(new_data))
        super().__init__(new_data)
    
    def unresym(self, regrouped_predictions):
        # Each group is a list of (prediction, trace, formula)
        out = []
        for group, (base_trace, base_formula) in zip(regrouped_predictions, self.base_dataset.data):
            unique_variables = sorted(set([ch for ch in base_trace if ch.isalpha()]))
            predictions = set()
            for item, perm in zip(group, permutations(self.aps, len(unique_variables))):
                # Undo the permutation
                translation_map = {new: original for original, new in zip(unique_variables, perm)}
                prediction = ''.join([translation_map.get(ch, ch) for ch in item[0]])
                predictions.add(prediction)
            item = {
                "trace": base_trace,
                "target": base_formula,
                "predictions": list(predictions),
                "pred_count": len(predictions),
                "perm_count": len(group),
                "ap_count": len(unique_variables),
            }
            if len(group) > 1:
                item["TRC"] = 1.0 - ((len(predictions)-1) / (len(group)-1))
            out.append(item)
        return out


def evaluate_model(model_path, args, load_model, get_gen_args):
    model = load_model(model_path, **vars(args))
    print("Loaded pretrained model:", model_path)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {param_count:_}")

    dataset_vocab = dataset.get_dataset_vocab(args, model.config)
    # Only input compatibility is required
    if not model.config.vocab.are_inputs_compatible(dataset_vocab):
        sys.exit("Dataset vocabulary is not compatible with the model")
    test_dataset = dataset.get_dataset(args, args.split, dataset.RawLTLDataset, max_samples=args.max_samples)
    resym_dataset = ResymbolizeDataset(test_dataset, dataset_vocab.aps)

    gen_args = get_gen_args(args)
    if args.syntax_enforcing:
        gen_args['syntax_enforcer'] = LTLSyntaxEnforcerConfig(model.config.vocab)

    result_dir = os.path.join(model_path, args.result_dir_name)
    os.makedirs(result_dir, exist_ok=True)

    predictions = model.generate_predictions(resym_dataset, args.max_length, gen_args)
    it = iter(predictions)
    regrouped = [[next(it) for _ in range(group_size)] for group_size in resym_dataset.group_sizes]

    # results = trace_check.evaluate_ltl(predictions, threads=args.eval_threads, timeout=args.eval_timeout)
    with open(os.path.join(result_dir, "raw_predictions.json"), 'w') as f:
        json.dump(regrouped, f, indent=4)
    
    unresymed = resym_dataset.unresym(regrouped)
    with open(os.path.join(result_dir, "predictions.json"), 'w') as f:
        json.dump(unresymed, f, indent=4)

    startlen = len(unresymed)
    unresymed = [item for item in unresymed if "TRC" in item]
    filtered = startlen - len(unresymed)
    if filtered > 0:
        print(filtered, "samples didn't contain any APs (only one permutation) and filtered out")

    all_trcs = [item["TRC"] for item in unresymed]
    ap_trcs = defaultdict(list)
    for item in unresymed:
        ap_count = item["ap_count"]
        ap_trcs[ap_count].append(item["TRC"])

    summary = {
        "TRC": describe_statistics(all_trcs, True),
    }
    summary |= {f"{k} AP TRC": describe_statistics(v, True) for k, v in ap_trcs.items()}
    print("SUMMARY:")
    pprint(summary)

    with open(os.path.join(result_dir, "predictions.json"), 'w') as f:
        json.dump(unresymed, f, indent=4)

    print("Generating and saving plots...")
    ap_trcs = dict(sorted(ap_trcs.items()))
    ap_trcs["All"] = all_trcs
    resym_plot("TCR Box Plot", ap_trcs, os.path.join(result_dir, "TCR-box.png"), violin=False)
    resym_plot("TCR Violin Plot", ap_trcs, os.path.join(result_dir, "TCR-violin.png"), violin=True)
    print("Done")


def resymbolize_eval(args, load_model, get_gen_args):
    init_plot_font()

    if args.result_dir_name is None:
        args.result_dir_name = "resym-" + get_result_dir_name(args)
        print("Result directory name:", args.result_dir_name)

    model_paths = glob(args.model_path)
    if not model_paths:
        print(f"No models found at {args.model_path}")
        return
    try:
        from natsort import natsorted
        model_paths = natsorted(model_paths)
    except ImportError:
        pass
    for model_path in model_paths:
        try:
            evaluate_model(model_path, args, load_model, get_gen_args)
        except Exception:
            trace = traceback.format_exc()
            print("Error during the evaluation of", model_path)
            print("Trace:")
            print(trace)
        print()


import matplotlib.pyplot as plt

def resym_plot(title, data, save_to, violin):
    figure, ax = plt.subplots(figsize=(7, 5))

    labels = [f'{k}\nn={len(v)}' for k, v in data.items()]
    xvalues = list(data.values())

    if violin:
        ax.violinplot(xvalues, showmeans=False, showmedians=True)
    else:
        ax.boxplot(xvalues)
    ax.set_title(title)

    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(labels))], labels=labels)
    ax.set_xlabel('AP count')
    ax.set_ylabel('Resymbolization Consistency')

    plt.show()
    figure.savefig(save_to, bbox_inches="tight", dpi=192)
