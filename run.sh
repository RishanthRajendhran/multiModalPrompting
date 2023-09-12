#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH -o outputs-%j

# export PYTHONPATH=/scratch/general/vast/u1419542/miniconda3/envs/cs6966Env/bin/python
# source /scratch/general/vast/u1419542/miniconda3/etc/profile.d/conda.sh
# conda activate cs6966Env

# wandb disabled 
# mkdir /scratch/general/vast/u1419542/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/u1419542/huggingface_cache"

OUT_DIR=/scratch/general/vast/u1419542/cs6966/assignment2/models
mkdir -p ${OUT_DIR}

python3 prompt.py --output_dir ${OUT_DIR}