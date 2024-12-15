#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <commands_file>"
    exit 1
fi

commands_file="$1"

while IFS= read -r wget_command; do
    sbatch --job-name=wget_job \
            --ntasks=1 \
            --cpus-per-task=1 \
            --time=02:00:00 \
            --mem=1G \
            --error=%j_%a_%N.err \
            --output=%j_%a_%N.out \
            --wrap="$wget_command"
done < "$commands_file"

