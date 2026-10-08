#!/bin/bash
#SBATCH --job-name=debug_mmed
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --account=def-arashmoh_gpu
#SBATCH --output=/home/gkianfar/scratch/Amin/concept/outputs/logs/debug_mmed_%j.out
#SBATCH --error=/home/gkianfar/scratch/Amin/concept/outputs/logs/debug_mmed_%j.err
module purge
module load gcc python/3.10 cuda/12.6 opencv/4.10.0
source /home/gkianfar/scratch/Amin/conceptvenv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis/checkpoint/hf_cache
export HF_HUB_CACHE=$HF_HOME
cd /home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis

python mmed_refiner.py
