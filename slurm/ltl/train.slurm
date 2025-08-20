#!/bin/bash
#SBATCH --time=1-00:00:00 # DD-HH:MM:SS
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20  # Cores
#SBATCH --gres=gpu:1

# Echo commands as they are executed
set -x

if [[ ! -n "$MODEL_PATH" ]]; then
	echo No MODEL_PATH, abort
	exit 1
fi
if [[ ! -n "$SEED" ]]; then
	echo No SEED, abort
	exit 1
fi

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python3 -m autoregltl.main --model-path=$MODEL_PATH --seed=$SEED train-ted "$@"

python3 -m autoregltl.main --model-path=$MODEL_PATH eval-ted --ds-name=ltl-35-10ap --beam-size=3
python3 -m autoregltl.main --model-path=$MODEL_PATH eval-ted --beam-size=3

python -m autoregltl.eval2da $MODEL_PATH


# for AP_COUNT in 3 4 5 6 7 8 9 10
# do
# 	python3 -m autoregltl.main --model-path=$MODEL_PATH --seed=42 resym-eval-ted --ds-name=ltl-each1k-10ap --vocab-aps=10 --split=val --max-samples=1000 --max-perm=120 --exact-aps=$AP_COUNT --beam-size=3 --eval-timeout=120 --result-dir-name="resym-${AP_COUNT}ap"
# done

echo "JOB DONE!!!"
