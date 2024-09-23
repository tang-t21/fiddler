#!/bin/bash

# Define an array of batch sizes
batch_sizes=(512)
ntoken=32
token_num=128
# Loop through each batch size
for batch_size in "${batch_sizes[@]}"; do
    echo "Running with batch size: $batch_size"
    python3 llama-bench.py --batch_size="${batch_size}" --token_num="$token_num" --n-token="$ntoken" --repeat 5
done
