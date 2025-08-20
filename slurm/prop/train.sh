#!/bin/bash

MODEL_NAME=$1
AP_COUNT=$2
export MODEL_PATH="models-prop/${AP_COUNT}ap/$MODEL_NAME"
full_model_path="~/deepltl/$MODEL_PATH"

PARAMS=(
	# Normally this should be like this, but separate enc/dec dims disallow merging vocabs
	# --d-embed-enc=128 --d-embed-dec=64
	# Also having 6 heads forces us to round 128 down to 126 = 6*21
	# But RoPE doesn't like 21 dims per head, so 22*6 = 132
	--d-embed-enc=132
	--num-heads=6 --d-ff=512 --num-layers=6
	# Train settings
	--batch-size=1024
	--epochs=64
	--val-max-samples=10000
	# Vocab
	--merge-tokens=all --merged-vocab
	--embed-scaling=sqrtd
	# Pos
	--tree-pos-enc
)

# Split MODEL_NAME by "-" and store in an array
IFS='-' read -r -a model_parts <<< "$MODEL_NAME"

# Process the first group: d005
first_group="${model_parts[0]}"

if [[ "$AP_COUNT" == "5" ]]; then
	if [[ "$first_group" == "0000" ]]; then
		PARAMS+=(--ds-name=ltl-35)
	else
		PARAMS+=(--ds-name=ltl-35-perturbed)
	fi
else
	PARAMS+=(--ds-name=ltl-35-"$AP_COUNT"ap)
fi

if [[ "$first_group" != "0000" ]]; then
	letter_part="${first_group:0:1}"  # Extract the first letter
	number_part="${first_group:1:3}"    # Extract the number part

	# Add --ap_embed based on the letter part
	if [[ "$letter_part" == "l" ]]; then
		second_number="${first_group:5}"
		PARAMS+=(--dynamic-aps)
		PARAMS+=(--ap_embed=learnable$number_part)
		PARAMS+=(--d_ap="$second_number")
	elif [[ "$letter_part" == "d" ]]; then
		PARAMS+=(--dynamic-aps)
		PARAMS+=(--ap_embed=diagbor)
		PARAMS+=(--d_ap="$number_part")
	elif [[ "$letter_part" == "r" ]]; then
		PARAMS+=(--dynamic-aps)
		PARAMS+=(--ap_embed=randn)
		PARAMS+=(--d_ap="$number_part")
	elif [[ "$letter_part" == "n" ]]; then
		PARAMS+=(--dynamic-aps)
		PARAMS+=(--ap_embed=nbor)
		PARAMS+=(--d_ap="$number_part")
	elif [[ "$letter_part" == "s" ]]; then
		PARAMS+=(--dynamic-aps)
		PARAMS+=(--shuffle-aps="$number_part")
	fi
fi

# Process the second group: rop
second_group="${model_parts[1]}"
if [[ "$second_group" == "rop" ]]; then
	PARAMS+=(--dec-pe=rope)
else
	PARAMS+=(--dec-pe=sinusoid)
fi

# Process the third group: bn1
third_group="${model_parts[2]}"
if [[ "$third_group" == "bn1" ]]; then
	PARAMS+=(--embed-base-normalization=l2)
	PARAMS+=(--embed-ap-normalization=l2)
else
	PARAMS+=(--embed-base-normalization=disabled)
	PARAMS+=(--embed-ap-normalization=disabled)
fi

# Process the fourth group: fn1
fourth_group="${model_parts[3]}"
if [[ "$fourth_group" == "fn1" ]]; then
	PARAMS+=(--embed-final-normalization=l2)
else
	PARAMS+=(--embed-final-normalization=disabled)
fi

# Process the fifth group: ada1
fifth_group="${model_parts[4]}"
if [[ "$fifth_group" == "ada1" ]]; then
	if [[ "$fourth_group" != "fn1" ]]; then
		echo "ada without fn, NOT ALLOWED"
		exit 1
	fi
	PARAMS+=(--feature-normalization=l2)
	PARAMS+=(--loss-fct=adacos)
fi

# Process the sixth group: s42 (extract the number)
sixth_group="${model_parts[5]}"
export SEED="${sixth_group:1}"  # Extract number after 's'

logfile="$full_model_path.out"
echo Logfile:
echo $logfile

if [ -d "$full_model_path" ]; then
	echo "Model does exist."
	exit 1
fi
if [ -d "$full_model_path.out" ]; then
	echo "Model does exist."
	exit 1
fi

#echo "${PARAMS[@]}"
sbatch -A etur17 -J $MODEL_NAME --output=$logfile --error=$logfile train.slurm "${PARAMS[@]}"
