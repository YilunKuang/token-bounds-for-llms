#!/bin/bash


list_of_models=("Llama-1-13b-E8P-2Bit"     "Llama-1-7b-E8PRVQ-3Bit"        "Llama-2-70b-chat-E8PRVQ-4Bit"
"Llama-1-13b-E8PRVQ-3Bit"  "Llama-1-7b-E8PRVQ-4Bit"        "Llama-2-70b-E8P-2Bit"
"Llama-1-13b-E8PRVQ-4Bit"  "Llama-2-13b-chat-E8P-2Bit"     "Llama-2-70b-E8PRVQ-3Bit"
"Llama-1-30b-E8P-2Bit"     "Llama-2-13b-chat-E8PRVQ-3Bit"  "Llama-2-70b-E8PRVQ-4Bit"
"Llama-1-30b-E8PRVQ-3Bit"  "Llama-2-13b-chat-E8PRVQ-4Bit"  "Llama-2-7b-chat-E8P-2Bit"
"Llama-1-30b-E8PRVQ-4Bit"  "Llama-2-13b-E8P-2Bit"          "Llama-2-7b-chat-E8PRVQ-3Bit"
"Llama-1-65b-E8P-2Bit"     "Llama-2-13b-E8PRVQ-3Bit"       "Llama-2-7b-chat-E8PRVQ-4Bit"
"Llama-1-65b-E8PRVQ-3Bit"  "Llama-2-13b-E8PRVQ-4Bit"       "Llama-2-7b-E8P-2Bit"
"Llama-1-65b-E8PRVQ-4Bit"  "Llama-2-70b-chat-E8P-2Bit"     "Llama-2-7b-E8PRVQ-3Bit"
"Llama-1-7b-E8P-2Bit"      "Llama-2-70b-chat-E8PRVQ-3Bit"  "Llama-2-7b-E8PRVQ-4Bit")

for curr_model in "${list_of_models[@]}"; do
        echo "$curr_model"

        sbatch --job-name=quip_llm360 \
                --nodes=1 \
                --ntasks-per-node=1 \
                --cpus-per-task=4 \
                --gres=gpu:a100:1 \
                --time=36:00:00 \
                --mem=64G \
                --error=%j_%a_%N.err \
                --output=%j_%a_%N.out \
                --wrap="singularity exec --nv --overlay /scratch/yk2516/singularity/overlay-25GB-500K-QUIP.ext3:ro /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c 'source /ext3/env.sh; conda activate base; cd /scratch/yk2516/repos/PAC_Bayes/token-bounds-llms; export PYTHONPATH=/scratch/yk2516/repos/PAC_Bayes/token-bounds-llms; python experiments/eval_bounds.py --config-file=/scratch/yk2516/repos/PAC_Bayes/token-bounds-llms/config/config_original_model_bounds.yaml --bounds.misc_extra_bits=2 --sublora.intrinsic_dim=0 --optimizer.learning_rate=0.0002 --model.best_checkpoint_path=/scratch/yk2516/repos/PAC_Bayes/token-bounds-llms/quip_model_checkpoints/\"$curr_model\" --model.init_from=relaxml/\"$curr_model\" --bounds.bound_type=token_level --bounds.bound_samples=10000 --bounds.use_quip=True --bounds.quip_model=relaxml/\"$curr_model\" --bounds.quip_model_cache_dir=/scratch/yk2516/cache --data.dataset_dir=None --data.dataset=llm360 --data.vocab_size=32000 --data.eot_token=2 --data.batch_size=1 --bounds.eval_batch_size=1'"

done
