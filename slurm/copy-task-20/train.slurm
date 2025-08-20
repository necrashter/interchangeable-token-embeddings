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

python3 -m autoregltl.main --model-path=$MODEL_PATH --seed=$SEED train-ted "$@"
python3 -m autoregltl.eval2da --test --max-aps=160 --max-length=160 --max-samples 20 $MODEL_PATH

echo "JOB DONE!!!"
