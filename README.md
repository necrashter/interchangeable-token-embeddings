# Interchangeable Token Embeddings

Repository for our ICML 2025 paper, [Interchangeable Token Embeddings for Extendable Vocabulary and Alpha-Equivalence](https://arxiv.org/abs/2410.17161).

## Requirements

In addition to PyTorch, HuggingFace Transformers and the LTL library [Spot](https://spot.lrde.epita.fr) is required. See the [download and installation instructions](https://spot.lrde.epita.fr/install.html) on their website.

The following commands will create a conda environment called `ltl` with the required packages:
```sh
conda create --name ltl python=3.10
conda activate ltl
conda install -c conda-forge spot
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

The master branch contains the LTL (linear temporal logic) task.
Switch to `prop-task` or `copy-task` branches for the propositional logic and copying with extensible vocabulary tasks, respectively.
Note that the dataset names remain consistent between LTL and propositional logic tasks, e.g., the default prop. logic dataset is called `ltl-35`, its 10 AP variant is `ltl-35-10ap`, etc.
To differentiate the datasets, in LTL, the datasets under `data` folder is used, whereas in prop. logic, `data-prop` folder is used.

## Datasets & Models

You can get the datasets and the trained models from HuggingFace:
* [Datasets](https://huggingface.co/datasets/necrashter/interchangeable-token-embeddings-datasets)
* [Models](https://huggingface.co/necrashter/interchangeable-token-embeddings)

## Usage

Before continuing, set the following environment variable for deterministic behavior:
```sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### Training

```sh
# LTL Generalization Models
python3 -m autoregltl.main --model-path=models/ltl-generalization/proposed --seed=42 train-ted --ds-name=ltl-35-perturbed --num-heads=8 --d-embed-enc=128 --d-ff=1024 --num-layers=8 --batch-size=768 --epochs=50 --val-max-samples=10000 --merge-tokens=all --merged-vocab --embed-scaling=sqrtd --dynamic-aps --d_ap=5 --ap_embed=diagbor --feature-normalization=l2 --loss-fct=adacos --tree-pos-enc --dec-pe=rope
python3 -m autoregltl.main --model-path=models/ltl-generalization/baseline train-ted --ds-name=ltl-35 --num-heads=8 --d-embed-enc=128 --d-ff=1024 --num-layers=8 --batch-size=768 --epochs=50 --val-max-samples=10000 --merge-tokens=all --merged-vocab --embed-scaling=sqrtd --feature-normalization=l2 --loss-fct=adacos --tree-pos-enc --dec-pe=rope
python3 -m autoregltl.main --model-path=models/ltl-generalization/full-vocab train-ted --ds-name=ltl-35-10ap --num-heads=8 --d-embed-enc=128 --d-ff=1024 --num-layers=8 --batch-size=768 --epochs=50 --val-max-samples=10000 --merge-tokens=all --merged-vocab --embed-scaling=sqrtd --feature-normalization=l2 --loss-fct=adacos --tree-pos-enc --dec-pe=rope
python3 -m autoregltl.main --model-path=models/ltl-generalization/alpha-renaming --seed=46 train-ted --num-heads=8 --d-embed-enc=128 --d-ff=1024 --num-layers=8 --batch-size=768 --epochs=50 --val-max-samples=10000 --merge-tokens=all --merged-vocab --embed-scaling=sqrtd --tree-pos-enc --ds-name=ltl-35-perturbed --dynamic-aps --shuffle-aps=010 --dec-pe=rope --embed-base-normalization=l2 --embed-ap-normalization=l2 --embed-final-normalization=l2 --feature-normalization=l2 --loss-fct=adacos
```
See [autoregltl/main.py](./autoregltl/main.py) for more command line arguments.
The [slurm](./slurm) folder contains helper scripts for training. The `train.sh` script for each task submits a new job using `train.slurm`.
You can learn more about training commands by inspecting these files.

Note that you are not expected to get the same results if your hardware or package versions are different.

### Evaluation

```sh
# Default validation & test set
python3 -m autoregltl.main --model-path=$MODEL_PATH eval-ted --beam-size=3
# The following corresponds to the evaluation section in the perturbation table.
python3 -m autoregltl.main --model-path=$MODEL_PATH eval-ted --beam-size=3 --split=test
# 10 AP validation set
python3 -m autoregltl.main --model-path=$MODEL_PATH eval-ted --ds-name=ltl-35-10ap --beam-size=3
```
Replace `$MODEL_PATH` with the model directory.
You can add `--max-samples=1000` to limit the number of traces.
Evaluating with these commands will create a results folder under the model directory.

Alpha-covariance evaluation on the 10 AP validation set:
```sh
for AP_COUNT in 3 4 5 6 7 8 9 10
do
	python3 -m autoregltl.main --model-path=$MODEL_PATH --seed=42 resym-eval-ted --ds-name=ltl-each1k-10ap --vocab-aps=10 --split=val --max-samples=1000 --max-perm=120 --exact-aps=$AP_COUNT --beam-size=3 --eval-timeout=120 --result-dir-name="resym-${AP_COUNT}ap"
done
```

Alpha-covariance evaluation on the 5 AP test set (perturbation table):
```sh
python3 -m autoregltl.main --model-path=$MODEL_PATH resym-eval-ted --ds-name=ltl-35 --split=test --max-samples=1000 --exact-aps=3 --beam-size=3 --eval-timeout=120 --result-dir-name="resym-ltl35-test1k-3ap-b3"
python3 -m autoregltl.main --model-path=$MODEL_PATH resym-eval-ted --ds-name=ltl-35 --split=test --max-samples=1000 --exact-aps=4 --beam-size=3 --eval-timeout=120 --result-dir-name="resym-ltl35-test1k-4ap-b3"
python3 -m autoregltl.main --model-path=$MODEL_PATH resym-eval-ted --ds-name=ltl-35 --split=test --max-samples=1000 --exact-aps=5 --beam-size=3 --eval-timeout=120 --result-dir-name="resym-ltl35-test1k-5ap-b3"
```

Heatmap evaluation:
```sh
python -m autoregltl.eval2da "$MODEL_PATH"
```

### Utils

There are various scripts in the `utils` and `notebooks` folders for creating perturbed datasets, figures, etc.

