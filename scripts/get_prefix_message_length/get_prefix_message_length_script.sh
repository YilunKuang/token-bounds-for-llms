#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <base_dir>"
    exit 1
fi

# Base directory containing the folders
base_dir="$1"

# List of folders to iterate over
folders=(
    "Llama-1-13b-E8PRVQ-3Bit" "Llama-1-30b-E8PRVQ-3Bit" "Llama-1-65b-E8PRVQ-3Bit"
    "Llama-1-13b-E8PRVQ-4Bit" "Llama-1-30b-E8PRVQ-4Bit" "Llama-1-65b-E8PRVQ-4Bit"
    "Llama-1-7b-E8PRVQ-3Bit" "Llama-1-7b-E8PRVQ-4Bit"
    "Llama-2-13b-chat-E8PRVQ-4Bit"
    "Llama-2-13b-E8PRVQ-4Bit"
    "Llama-2-70b-chat-E8PRVQ-4Bit"
    "Llama-2-70b-E8PRVQ-4Bit"
    "Llama-2-7b-chat-E8PRVQ-3Bit" "Llama-2-7b-chat-E8PRVQ-4Bit"
    "Llama-2-7b-E8PRVQ-3Bit" "Llama-2-7b-E8PRVQ-4Bit"
)

for folder in "${folders[@]}"; do
    echo "Submitting job for folder: $folder"
    
    sbatch --job-name="process_$folder" \
           --ntasks=1 \
           --cpus-per-task=1 \
           --time=02:00:00 \
           --mem=36G \
           --output="$base_dir/$folder/slurm-%j.out" \
           --error="$base_dir/$folder/slurm-%j.err" \
           --wrap="
cd $base_dir/$folder

# Check if there is any file ending with .gz
if ls *.gz 1> /dev/null 2>&1; then
    echo 'Found .gz file in $folder, skipping...'
    exit 0
fi

# Check if there are files starting with 'model'
model_files=(\$(ls model*.safetensors 2> /dev/null))

if [ \${#model_files[@]} -eq 0 ]; then
    echo 'No model files found in $folder'
elif [ \${#model_files[@]} -eq 1 ]; then
    echo 'Single model file found: \${model_files[0]}'
    gzip -c \${model_files[0]} > \${model_files[0]}.gz
else
    echo 'Multiple model files found: \${model_files[@]}'
    tar -czvf combined_file.tar.gz \${model_files[@]}
fi
"
done
