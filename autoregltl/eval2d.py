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

device = torch.device('cuda')

def triangular_mean(matrix):
    """
    Calculates the upper triangular mean of a matrix.
    """
    mindim = min(matrix.size())
    lowtri = mindim * (mindim-1) / 2
    return matrix.sum().item() / (matrix.numel() - lowtri)


@torch.no_grad
def eval2d(
        model_path,
        max_samples,
        min_aps=3,
        max_aps=30,
        min_length=3,
        max_length=30,
        repeat_count=1,
        figsize=(6, 5),
        eval_ds="cpy-eval",
        gen_args=None,
        output = "eval2d.pkl",
        save_all_predictions=False,
    ):
    if gen_args is None:
        gen_args = dict(
            alpha=1.0,
            beam_size=1,
            gen_batch_size=512,
        )
    save_loc = os.path.join(model_path, output)
    if save_all_predictions:
        save_loc_dir = os.path.join(model_path, output + ".d")
        os.makedirs(save_loc_dir, exist_ok=True)
    model = ted.load_model(model_path, device)
    model.eval()

    if not model.config.vocab.dynamic_aps and max_aps > len(model.config.vocab.aps):
        max_aps = len(model.config.vocab.aps)

    if model.config.merged_embedder.ap_embed == "diagbor":
        max_aps = min(max_aps, 2**model.config.merged_embedder.d_ap)
    elif model.config.merged_embedder.ap_embed == "nbor":
        max_aps = min(max_aps, 3**model.config.merged_embedder.d_ap - 1)

    datasets = {}
    all_pairs = []
    for ap in tqdm(range(3, max_aps+1), desc="Datasets"):
        sizes = []
        datas = []
        for l in range(ap, max_length+1):
            data = dataset.RawLTLDataset(
                f"data/{eval_ds}/{ap}-{l}.txt", -1, -1,
                max_samples=max_samples,
                print_filtered=False,
            ).data
            sizes.append(len(data))
            datas += data
            all_pairs += data
        test_dataset = dataset.SeqDataset(datas)
        datasets[ap] = (test_dataset, sizes)

    all_dataset = dataset.EncDecLTLDataset(
        filename=None,
        vocab=model.config.vocab,
        max_formula_length=None,
        max_trace_length=None,
        pairs=all_pairs,
    )
    crossent = torch.nn.CrossEntropyLoss(reduction='sum')

    evals = []
    for repetition in tqdm(range(repeat_count), desc="Reps"):
        if model.config.vocab.dynamic_aps:
            model.config.vocab.aps = CHARS[:max_aps]
        model.merged_embedder.prepare()

        w_matrix = model.merged_embedder.w.detach().cpu()
        all_predictions = {}

        # Compute cross entropy loss on all_dataset
        # Initalize dataloader
        dataloader = torch.utils.data.DataLoader(
            all_dataset,
            batch_size=512,
            shuffle=False,
            collate_fn=dataset.EncDecLTLCollator(),
        )
        loss = 0
        for inputs in dataloader:
            logits = model(inputs["input_ids"].to(device), inputs["target_ids"].to(device))
            labels = torch.where(inputs["target_ids"] == model.pad_id, -100, inputs["target_ids"]).to(device)
            loss += crossent(logits.view(-1, logits.size(-1)), labels.view(-1)).item() / len(all_pairs)

        correct_matrix = torch.zeros(max_aps - min_aps + 1, max_length - min_length + 1)
        editdist_matrix = torch.zeros(max_aps - min_aps + 1, max_length - min_length + 1)
        for apcount in tqdm(list(range(3, max_aps+1))[::-1], desc="APs", leave=False):
            if model.config.vocab.dynamic_aps:
                model.config.vocab.aps = CHARS[:apcount]
                model.merged_embedder.shrink_w()

            test_dataset, sizes = datasets[apcount]
            cum_preds = model.generate_predictions(
                test_dataset,
                max_length=max_length*2,
                gen_args=gen_args,
                leave_tqdm=False,
                prepare_embedder=False,  # generate_predictions should NOT re-prep embedder
            )
            for l, size in zip(range(apcount, max_length+1), sizes):
                predictions, cum_preds = cum_preds[:size], cum_preds[size:]
                correct = sum([a == b for a, b in predictions])
                all_predictions[(apcount, l)] = predictions
                correct_matrix[apcount-min_aps, l-min_length] += correct / max_samples
                t = torch.tensor([editdistance.eval(*a) for a in predictions], dtype=torch.float64)
                editdist_matrix[apcount-min_aps, l-min_length] = t.mean()
        
        evals.append({
            "correct_matrix": correct_matrix,
            "editdist_matrix": editdist_matrix,
            "w_matrix": w_matrix,
            "correct": triangular_mean(correct_matrix),
            "editdist": triangular_mean(editdist_matrix),
            "loss": loss,
        })
        if save_all_predictions:
            with open(os.path.join(save_loc_dir, f"all_predictions{repetition}.pkl"), 'wb') as f:
                pickle.dump(all_predictions, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Sort evals by editdist
    evals = sorted(evals, key=lambda x: x["editdist"])

    # Compute average correct and editdist
    correct = np.mean([e["correct"] for e in evals])
    editdist = np.mean([e["editdist"] for e in evals])
    print(f"Correct: {correct}")
    print(f"Editdist: {editdist}")
    results = {
        "evals": evals,
        "correct": correct,
        "editdist": editdist,
    }

    results |= {
        "min_aps": min_aps,
        "min_length": min_length,
        "max_aps": max_aps,
        "max_length": max_length,
        "repeat_count": repeat_count,
        "eval_ds": eval_ds,
    }
    # SAVE
    with open(save_loc, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    ds_name = None
    try:
        # Determine model's training dataset
        with open(os.path.join(model_path, "command-log.txt"), 'r') as f:
            # Find the line that starts with '    "ds_name": '
            for line in f:
                if line.startswith('    "ds_name": '):
                    ds_name = line.split('"')[-2]
                    ds_info = ds_name.split('-')
                    train_minlen = int(ds_info[1])
                    train_maxlen = int(ds_info[2])
                    train_maxap = int(ds_info[3][:-2])
                    break
    except:
        print("Failed to determine training dataset")

    # Plot
    editdistmat = torch.stack([e["editdist_matrix"] for e in evals])
    editdistmat = torch.mean(editdistmat, dim=0)
    apticks = list(range(min_aps, max_aps+1))
    lenticks = list(range(min_length, max_length+1))
    # plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
    # plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
    # plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    # plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    mpl.rcParams['hatch.linewidth'] = 10.0
    mpl.rcParams['hatch.color'] = "#db2114"
    fig = plt.figure(figsize=figsize)
    ax = sn.heatmap(
        editdistmat,
        yticklabels=apticks, xticklabels=lenticks,
        cmap=sn.cm.rocket_r,
        vmax=max(max_length, editdistmat.max().item()),
    )
    if ds_name is not None:
        # Denote training area
        ax.add_patch(mpl.patches.Rectangle((train_minlen - min_length, 0), train_maxlen - train_minlen + 1, train_maxap - min_aps + 1, fill=False, edgecolor='#00cc1f', lw=2))
    for y in range(1, max_aps - min_aps + 1):
        ax.add_patch(mpl.patches.Rectangle((0, y), y, 1, hatch='/', facecolor='#9a1a0f', edgecolor="#550000", lw=0))
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    ax.set_ylabel("Vocabulary size")
    ax.set_xlabel("Length")
    plt.savefig(os.path.join(model_path, "eval2d.png"), bbox_inches="tight", dpi=192, pad_inches=0.02)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, nargs='+')
    parser.add_argument('--max-samples', type=int, default=1000)
    parser.add_argument('--min-aps', type=int, default=3)
    parser.add_argument('--max-aps', type=int, default=30)
    parser.add_argument('--min-length', type=int, default=3)
    parser.add_argument('--max-length', type=int, default=30)
    parser.add_argument('--repeat-count', type=int, default=1)
    parser.add_argument('--figsize', type=str, default="(6,5)")
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')
    parser.add_argument('--output', type=str, default="eval2d.pkl")
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
                args.max_samples,
                args.min_aps,
                args.max_aps,
                args.min_length,
                args.max_length,
                args.repeat_count,
                eval(args.figsize),
                eval_ds="cpy-eval-test" if args.test else "cpy-eval",
                output=args.output,
            )
        except Exception as e:
            print("Error:", e)