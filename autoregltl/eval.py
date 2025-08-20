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

from autoregltl.ltl.enforcer import LTLSyntaxEnforcerConfig
from autoregltl import dataset
from autoregltl.ltl import trace_check
from autoregltl.utils import describe_statistics, tictoc_histogram, init_plot_font


def analyze_ltl_results(results, args, result_dir):
    analysis = trace_check.analyze_results(results)
    res = trace_check.per_size_analysis(analysis, save_analysis=os.path.join(result_dir, "size_hist"))
    total = len(results)
    res["correct"] = res.get("exact match", 0) + res.get("semantically correct", 0) + res.get("equivalent", 0)
    # For ordering headers in res
    order = defaultdict(lambda: 100)
    for i, e in enumerate(["correct", "exact match", "equivalent", "semantically correct", "incorrect"]):
        order[e] = i
    # Create summary string
    summary = ["EVALUATION SUMMARY"]
    for key, count in sorted(res.items(), key=lambda pair: order[pair[0]]):
        summary.append(f"{key.capitalize()}: {count}/{total}, {count / total * 100:f}%")
    summary = "\n".join(summary)

    print()
    print(summary)
    with open(os.path.join(result_dir, "summary.txt"), 'w') as f:
        f.write("Command Line Arguments:\n")
        f.write(" ".join(sys.argv[1:]))
        f.write("\n\n")
        f.write(summary)
        f.write("\n")

    with open(os.path.join(result_dir, "summary.json"), 'w') as f:
        json.dump(res, f, indent=4)


def evaluate_model(model_path, model, args, get_gen_args):
    dataset_vocab = dataset.get_dataset_vocab(args, model.config)
    # Only input compatibility is required
    if not model.config.vocab.are_inputs_compatible(dataset_vocab):
        sys.exit("Dataset vocabulary is not compatible with the model")
    test_dataset = dataset.get_dataset(args, args.split, dataset.RawLTLDataset, max_samples=args.max_samples)

    gen_args = get_gen_args(args)

    result_dir = os.path.join(model_path, args.result_dir_name)
    os.makedirs(result_dir, exist_ok=True)

    predictions = model.generate_predictions(test_dataset, args.max_length, gen_args)
    results = trace_check.evaluate_ltl(predictions, threads=args.eval_threads, timeout=args.eval_timeout, equivalence_method=args.equivalence)
    with open(os.path.join(result_dir, "evaluation.json"), 'w') as f:
        json.dump(results, f, indent=4)
    res = defaultdict(int)
    for result in results:
        res[result["result"]] += 1
    summary = {k: res.get(k, 0) for k in ["semantically correct", "exact match", "equivalent", "incorrect", "invalid", "timeout"]}
    print(summary)
    # analyze_ltl_results(results, args, result_dir)



def get_result_dir_name(args):
    out = f"{args.ds_name}-"
    if args.max_samples is not None:
        number = args.max_samples
        if number >= 1000:
            number = str(number // 1000) + 'k'
        out += f"{args.split}{number}"
    else:
        out += args.split

    if args.beam_size is not None:
        out += f"-b{args.beam_size}"

    return out


def evaluate(args, load_model, get_gen_args):
    try:
        from natsort import natsorted
    except ImportError:
        print("Install natsort package to sort glob results: pip install natsort")
        natsorted = lambda x: x

    init_plot_font()

    ds_names = args.ds_name
    ds_names = glob(os.path.join(args.data_dir, args.ds_name))
    if len(ds_names) > 1:
        ds_names = natsorted([os.path.basename(p) for p in ds_names])
        print("Found", len(ds_names), "datasets matching the pattern")
    else:
        ds_names = [args.ds_name]
    result_dir_name = args.result_dir_name

    model_paths = glob(args.model_path)
    if not model_paths:
        print(f"No models found at {args.model_path}")
        return

    for model_path in natsorted(model_paths):
        try:
            model = load_model(model_path, **vars(args))
            model.eval()
            print("Loaded pretrained model:", model_path)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of parameters: {param_count:_}")

            for ds_name in ds_names:
                print()
                print("Dataset name:", ds_name)
                args.ds_name = ds_name
                if result_dir_name is None:
                    args.result_dir_name = get_result_dir_name(args)
                elif len(ds_names) == 1:
                    args.result_dir_name = result_dir_name
                else:
                    # Prepend dataset name so that they don't clash
                    args.result_dir_name = ds_name + "-" + result_dir_name
                args.result_dir_name = os.path.join("results", args.result_dir_name)
                print("Result directory name:", args.result_dir_name)

                evaluate_model(model_path, model, args, get_gen_args)
        except Exception:
            trace = traceback.format_exc()
            print("Error during the evaluation of", model_path)
            print("Trace:")
            print(trace)
        print()