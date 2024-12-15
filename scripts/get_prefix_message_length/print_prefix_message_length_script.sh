#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <base_dir>"
    exit 1
fi

# Base directory containing the folders
base_dir="$1"

# List of folders to iterate over
folders=(
    "Llama-1-13b-E8PRVQ-4Bit" "Llama-1-30b-E8PRVQ-4Bit" "Llama-1-65b-E8PRVQ-4Bit"
    "Llama-1-7b-E8PRVQ-4Bit"
    "Llama-2-13b-chat-E8PRVQ-4Bit"
    "Llama-2-13b-E8PRVQ-4Bit"
    "Llama-2-70b-chat-E8PRVQ-4Bit"
    "Llama-2-70b-E8PRVQ-4Bit"
    "Llama-2-7b-chat-E8PRVQ-4Bit"
    "Llama-2-7b-E8PRVQ-4Bit"
)

for folder in "${folders[@]}"; do
    echo "Processing folder: $folder"
    cd "$base_dir/$folder" || { echo "Failed to enter $folder"; continue; }

    # Print the size of each .gz file
    for gz_file in *.gz; do
        if [ -f "$gz_file" ]; then
            size=$(ls -l "$gz_file" | awk '{print $5}')
            echo "Size of $gz_file: $size bytes"
        fi
    done

    cd "$base_dir"
done
