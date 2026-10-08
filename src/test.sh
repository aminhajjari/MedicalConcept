#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --account=def-arashmoh_gpu
#SBATCH --output=/home/gkianfar/scratch/Amin/concept/outputs/logs/smoke_%j.out
#SBATCH --error=/home/gkianfar/scratch/Amin/concept/outputs/logs/smoke_%j.err

set -e
module purge
module load gcc python/3.10 cuda/12.6 opencv/4.10.0
source /home/gkianfar/scratch/Amin/conceptvenv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis/checkpoint/hf_cache
export HF_HUB_CACHE=$HF_HOME

PROJECT_PATH="/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis"
MMED_CKPT="$PROJECT_PATH/checkpoint/MMed-Llama-3-8B-EnIns"
MISTRAL_CKPT="$PROJECT_PATH/checkpoint/Mistral-7B-Instruct"
cd $PROJECT_PATH

# results symlink so outputs land where the eval scripts expect
mkdir -p /home/gkianfar/scratch/Amin/concept/outputs/results
[ -e results ] || ln -s /home/gkianfar/scratch/Amin/concept/outputs/results results

echo "===== 1) GENERATION + confidence-aware refinement (Mistral refiner) ====="
python run_x_to_c_to_y.py --dataset PH2 --split 0 --model Explicd \
    --concept_extractor Explicd --generate_concepts --predict_for_train_set \
    --data_path data --refiner mistral --margin_threshold 0.2

echo "===== 2) CLASSIFIER PATH — MMed (the chat-template risk) ====="
python run_x_to_c_to_y.py --dataset PH2 --split 0 --concept_extractor Explicd \
    --llm MMed --ckpt $MMED_CKPT --n_demos 0 --refiner mistral

echo "===== 3) CLASSIFIER PATH — Mistral + 1-shot (tests new demo format) ====="
python run_x_to_c_to_y.py --dataset PH2 --split 0 --concept_extractor Explicd \
    --llm Mistral --ckpt $MISTRAL_CKPT --use_demos --n_demos 1 --refiner mistral

echo "===== SMOKE TEST PASSED ====="
