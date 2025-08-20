import traceback
import torch
import pickle, os
import editdistance
import numpy as np
from autoregltl import ted, dataset
from autoregltl.ltl.chars import CHARS
import random

from tqdm.auto import tqdm
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib as mpl
from autoregltl.ltl import trace_check

device = torch.device('cuda')

redgreen = mpl.colors.LinearSegmentedColormap(
    "redgreen",
    {
        'red': (
            (0.0, 1.0, 1.0),
            (0.15873*3, 1.0, 1.0),
            (0.174603*3, 0.96875, 0.96875),
            (1.0, 0.0, 0.0),
        ),
        'green': (
            (0.0, 0.0, 0.0),
            (0.15873*3, 0.9375, 0.9375),
            (0.174603*3, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ),
        'blue': (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
    }
)

@torch.no_grad
def eval2d(
        model_path,
        eval_ds,
        repeat_count=10,
        figsize=(6, 5),
        gen_args=None,
        output = "eval2da1.pkl",
    ):
    if gen_args is None:
        gen_args = dict(
            alpha=1.0,
            beam_size=3,
            gen_batch_size=512,
        )
    save_loc = os.path.join(model_path, output)
    model = ted.load_model(model_path, device)
    model.eval()

    with open(eval_ds, 'rb') as f:
        dsdict = pickle.load(f)
    
    min_aps = min([i[0] for i in dsdict.keys()])
    max_aps = max([i[0] for i in dsdict.keys()])
    min_length = min([i[1] for i in dsdict.keys()])
    max_length = max([i[1] for i in dsdict.keys()])
    
    if not model.config.vocab.dynamic_aps:
        max_aps = min(len(model.config.vocab.aps), max_aps)

    datasets = {}
    all_pairs = []
    for ap in range(min_aps, max_aps+1):
        sizes = []
        datas = []
        for l in range(min_length, max_length+1):
            data = dsdict.get((ap, l), [])
            sizes.append(len(data))
            datas += data
            all_pairs += data
        test_dataset = dataset.SeqDataset(datas)
        datasets[ap] = (test_dataset, sizes)

    print("All pairs:", len(all_pairs))
    all_dataset = dataset.EncDecLTLDataset(
        filename=None,
        vocab=model.config.vocab,
        max_formula_length=None,
        max_trace_length=None,
        tree_pos_enc=model.config.tree_pos_enc,
        pairs=all_pairs,
    )

    filedict = {}
    if model.config.vocab.dynamic_aps:
        model.config.vocab.aps = CHARS[:max_aps]
        median_w_out = model.set_median_w(all_dataset, repeat_count=repeat_count)
        filedict |= median_w_out
    elif (merged_embedder := getattr(model, "merged_embedder", None)):
        merged_embedder.prepare()

    correct_matrix = torch.zeros(max_aps + 1, max_length)
    count_matrix = torch.zeros(max_aps + 1, max_length)
    all_results = {}
    for apcount in tqdm(list(range(min_aps, max_aps+1))[::-1], desc="APs"):
        if model.config.vocab.dynamic_aps:
            model.config.vocab.aps = CHARS[:apcount]
            model.merged_embedder.shrink_w()

        test_dataset, sizes = datasets[apcount]
        cum_preds = model.generate_predictions(
            test_dataset,
            max_length=128,
            gen_args=gen_args,
            leave_tqdm=False,
            prepare_embedder=False,  # generate_predictions should NOT re-prep embedder
        )
        cum_results = trace_check.evaluate_ltl(cum_preds, timeout=30, leave_tqdm=False)
        for l, size in zip(range(min_length, max_length+1), sizes):
            results, cum_results = cum_results[:size], cum_results[size:]
            correct = 0
            for r in results:
                if r['result'] == 'semantically correct' or r['result'] == 'exact match':
                    correct += 1
            all_results[(apcount, l)] = results
            correct_matrix[apcount, l-1] += correct
            count_matrix[apcount, l-1] += len(results)
    
    filedict |= {
        "correct_matrix": correct_matrix,
        "count_matrix": count_matrix,
        "correct": correct_matrix.sum().item(),
        "count": count_matrix.sum().item(),
        "repeat_count": repeat_count,
        "eval_ds": eval_ds,
        "all_results": all_results,
    }
    print("Correct:", filedict["correct"])
    print("Count:", filedict["count"])
    print("Correct ratio:", filedict["correct"]/filedict["count"])

    # SAVE
    with open(save_loc, 'wb') as f:
        pickle.dump(filedict, f, protocol=pickle.HIGHEST_PROTOCOL)

    sample_rate = count_matrix / 100.0
    eval_results = torch.where(count_matrix > 0, correct_matrix / count_matrix, 0.0)
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    common_kwargs = dict(
        aspect='auto',
    )
    ax.imshow(eval_results, cmap=redgreen, vmin=0.0, vmax=1.0, **common_kwargs)
    # Plotting the modulus array as the 'value' part
    black = torch.zeros(max_aps+1, max_length, 4)
    black[:, :, -1] = 1.0 - sample_rate
    #black[:, :, -1] = torch.where(sample_rate > 0, 0.0, 1.0)
    ax.imshow(black, **common_kwargs)

    ax.set_ylabel("AP count")
    ax.set_xlabel("Formula length")
    xticks = [1, 10, 20, 30, 40, 50]
    ax.set_xticks([i -1 for i in xticks], xticks)
    # # 35 is not actually inclusive
    # ax.add_patch(mpl.patches.Rectangle((-0.5, -0.5), 35, 5+1, fill=False, edgecolor='white', lw=2))

    fig.colorbar(plt.cm.ScalarMappable(cmap=redgreen), ax=ax)
    plt.savefig(save_loc + ".png", bbox_inches="tight", dpi=192, pad_inches=0.02)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, nargs='+')
    parser.add_argument('--repeat-count', type=int, default=10)
    parser.add_argument('--figsize', type=str, default="(6,3)")
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')
    parser.add_argument('--input', type=str, default="data/eval2d-10ap.pkl")
    parser.add_argument('--output', type=str, default="eval2da1.pkl")
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)

    for model_path in args.model_path:
        print("Evaluating:", model_path)
        try:
            eval2d(
                model_path,
                repeat_count=args.repeat_count,
                figsize=eval(args.figsize),
                eval_ds=args.input,
                output=args.output,
            )
        except Exception as e:
            print("Error:")
            traceback.print_exc()