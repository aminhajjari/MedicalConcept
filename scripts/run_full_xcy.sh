#!/bin/bash
#SBATCH --job-name=xcy_fullexp
#SBATCH --account=def-arashmoh_gpu
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G
#SBATCH --output=/home/gkianfar/scratch/Amin/Skin/outputs/logs/xcy_%j.out
#SBATCH --error=/home/gkianfar/scratch/Amin/Skin/outputs/logs/xcy_%j.err

set -euo pipefail

echo "========================================="
echo "  x -> c -> y PIPELINE — ZERO + FEW SHOT"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "========================================="

# ---------- Paths ----------
PROJECT_PATH="/home/gkianfar/scratch/Amin/Skin/MedicalConcept"
DATA_PATH="$PROJECT_PATH/data"
CHECKPOINT_PATH="$PROJECT_PATH/checkpoint"

OUTPUT_BASE="/home/gkianfar/scratch/Amin/Skin/outputs"
RESULTS_BASE="$OUTPUT_BASE/results"
LOG_DIR="$OUTPUT_BASE/logs"

# Reuse the existing virtual environment
VENV_PATH="/home/gkianfar/scratch/Amin/conceptvenv311"

# ---------- Environment ----------
module purge
module load gcc python/3.11.5 opencv/4.13.0 arrow

source "$VENV_PATH/bin/activate"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TIMM_FUSED_ATTN=0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME="$CHECKPOINT_PATH/hf_cache"
export HF_HUB_CACHE="$HF_HOME"

export PYTHONWARNINGS="ignore"

# ---------- Model checkpoints ----------
MMED_CKPT="$CHECKPOINT_PATH/MMed-Llama-3-8B-EnIns"
MISTRAL_CKPT="$CHECKPOINT_PATH/Mistral-7B-Instruct"
QWEN_CKPT="$CHECKPOINT_PATH/Qwen2.5-72B-Instruct"
MEDGEMMA_CKPT="$CHECKPOINT_PATH/MedGemma-4b-it"

declare -A CKPT=(
    [MMed]="$MMED_CKPT"
    [Mistral]="$MISTRAL_CKPT"
    [Qwen]="$QWEN_CKPT"
    [MedGemma]="$MEDGEMMA_CKPT"
)

REFINERS=(rule mistral mmed)
CLASSIFIERS=(MMed Mistral Qwen MedGemma)

# ---------- Validate paths before starting ----------
cd "$PROJECT_PATH"

for path in "$DATA_PATH" "$CHECKPOINT_PATH" "$VENV_PATH/bin/python"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: Required path does not exist: $path" >&2
        exit 1
    fi
done

for model_path in "${CKPT[@]}"; do
    if [[ ! -d "$model_path" ]]; then
        echo "ERROR: Model checkpoint directory not found: $model_path" >&2
        exit 1
    fi
done

mkdir -p "$RESULTS_BASE/concept_prediction"
mkdir -p "$RESULTS_BASE/label_prediction"

# Use the new output directory for relative results/ paths.
if [[ -L results ]]; then
    rm results
elif [[ -e results ]]; then
    echo "ERROR: $PROJECT_PATH/results exists and is not a symlink." >&2
    echo "Move or inspect it before running this experiment." >&2
    exit 1
fi

ln -s "$RESULTS_BASE" results

echo "Project:      $PROJECT_PATH"
echo "Data:         $DATA_PATH"
echo "Checkpoints:  $CHECKPOINT_PATH"
echo "Results:      $RESULTS_BASE"
echo "Virtual env:  $VENV_PATH"
echo "Python:       $(which python)"
python --version

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# ---------- Run one pipeline stage ----------
run_stage() {
    echo ""
    echo ">>> $(date '+%H:%M:%S') python run_x_to_c_to_y.py $*"

    if ! python run_x_to_c_to_y.py "$@" 1>/dev/null; then
        echo "ERROR: Stage failed: $*" >&2
        exit 1
    fi

    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 2
}

# ============================================================
# STEP 1: GENERATE CONCEPTS (x -> c)
# ============================================================

echo ""
echo "========== STEP 1: Generate Concepts =========="

for refiner in rule mistral mmed; do
    for split in 0 1 2 3 4; do
        run_stage --dataset PH2 --split "$split" \
            --model Explicd --concept_extractor Explicd \
            --generate_concepts --predict_for_train_set \
            --data_path "$DATA_PATH" \
            --refiner "$refiner"
    done
    echo "PH2 refiner=$refiner done"
done

for dataset in Derm7pt HAM10000; do
    for refiner in rule mistral mmed; do
        run_stage --dataset "$dataset" \
            --model Explicd --concept_extractor Explicd \
            --generate_concepts --predict_for_train_set \
            --data_path "$DATA_PATH" \
            --refiner "$refiner"
        echo "$dataset refiner=$refiner done"
    done
done

echo "STEP 1 complete"

# ============================================================
# STEP 2: ZERO-SHOT + FEW-SHOT CLASSIFICATION
# ============================================================

echo ""
echo "========== STEP 2: Classification =========="

for n_shots in 0 1 2 4 8; do
    if [[ "$n_shots" -eq 0 ]]; then
        DEMOS_FLAG=()
        HAM_DEMOS=0
    else
        DEMOS_FLAG=(--use_demos)
        HAM_DEMOS=$(( n_shots > 2 ? 2 : n_shots ))
    fi

    for clf in "${CLASSIFIERS[@]}"; do
        run_stage --dataset PH2 --concept_extractor Explicd \
            --llm "$clf" --ckpt "${CKPT[$clf]}" \
            "${DEMOS_FLAG[@]}" --n_demos "$n_shots" \
            --refiner_list "${REFINERS[@]}" \
            --split_list 0 1 2 3 4

        run_stage --dataset Derm7pt --concept_extractor Explicd \
            --llm "$clf" --ckpt "${CKPT[$clf]}" \
            "${DEMOS_FLAG[@]}" --n_demos "$n_shots" \
            --refiner_list "${REFINERS[@]}"

        run_stage --dataset HAM10000 --concept_extractor Explicd \
            --llm "$clf" --ckpt "${CKPT[$clf]}" \
            "${DEMOS_FLAG[@]}" --n_demos "$HAM_DEMOS" \
            --refiner_list "${REFINERS[@]}"
    done

    echo "${n_shots}-shot classification done"
done

echo "STEP 2 complete"

# ============================================================
# STEP 2b: RULE-REFINED CONCEPT ABLATION
# ============================================================

echo ""
echo "========== STEP 2b: Rule Ablation =========="

RULE_D7="results/concept_prediction/Derm7pt_dermatology_reports_generated_by_Explicd_refiner_rule_raw_values_False.csv"
RULE_HAM="results/concept_prediction/HAM10000_dermatology_reports_generated_by_Explicd_refiner_rule_raw_values_False.csv"

for n_shots in 0 1 2; do
    if [[ "$n_shots" -eq 0 ]]; then
        DEMOS_FLAG=()
    else
        DEMOS_FLAG=(--use_demos)
    fi

    run_stage --dataset Derm7pt --concept_extractor Explicd \
        --report_path "$RULE_D7" --llm MMed \
        --ckpt "$MMED_CKPT" "${DEMOS_FLAG[@]}" \
        --n_demos "$n_shots" --refiner rule

    run_stage --dataset Derm7pt --concept_extractor Explicd \
        --report_path "$RULE_D7" --llm Mistral \
        --ckpt "$MISTRAL_CKPT" "${DEMOS_FLAG[@]}" \
        --n_demos "$n_shots" --refiner rule

    run_stage --dataset HAM10000 --concept_extractor Explicd \
        --report_path "$RULE_HAM" --llm MMed \
        --ckpt "$MMED_CKPT" "${DEMOS_FLAG[@]}" \
        --n_demos "$n_shots" --refiner rule

    run_stage --dataset HAM10000 --concept_extractor Explicd \
        --report_path "$RULE_HAM" --llm Mistral \
        --ckpt "$MISTRAL_CKPT" "${DEMOS_FLAG[@]}" \
        --n_demos "$n_shots" --refiner rule

    echo "Ablation ${n_shots}-shot done"
done

echo "STEP 2b complete"

# ============================================================
# STEP 2c: RANDOM-DEMO BASELINE
# ============================================================

echo ""
echo "========== STEP 2c: Random Demo Baseline =========="

for n_shots in 1 2; do
    for clf in "${CLASSIFIERS[@]}"; do
        run_stage --dataset PH2 --concept_extractor Explicd \
            --llm "$clf" --ckpt "${CKPT[$clf]}" \
            --use_demos --n_demos "$n_shots" \
            --refiner_list "${REFINERS[@]}" \
            --split_list 0 1 2 3 4 --random_demos

        run_stage --dataset Derm7pt --concept_extractor Explicd \
            --llm "$clf" --ckpt "${CKPT[$clf]}" \
            --use_demos --n_demos "$n_shots" \
            --refiner_list "${REFINERS[@]}" --random_demos

        run_stage --dataset HAM10000 --concept_extractor Explicd \
            --llm "$clf" --ckpt "${CKPT[$clf]}" \
            --use_demos --n_demos "$n_shots" \
            --refiner_list "${REFINERS[@]}" --random_demos
    done

    echo "Random baseline ${n_shots}-shot done"
done

# ============================================================
# STEP 2d: MULTI-EXPERT ENSEMBLE
# ============================================================

echo ""
echo "========== STEP 2d: Multi-Expert Ensemble =========="

python multi_expert_classifier.py \
    --experts MMed Mistral MedGemma Qwen \
    --sweep 0:rices 1:rices 2:rices 4:rices 8:rices 1:random 2:random \
    --tie melanoma \
    --prob_col p_melanoma

# ============================================================
# STEP 3: EVALUATION TABLES
# ============================================================

echo ""
echo "========== STEP 3: Results Tables =========="

python evaluate_results.py
python evaluate_fewshot_results.py
python analyze_refinement.py

echo ""
echo "========================================="
echo "  PIPELINE FINISHED"
echo "  Finished: $(date)"
echo "========================================="
