#!/bin/bash
#SBATCH --job-name tokenize-les_faits-v2 # Name for your job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --time=07:30:00
#SBATCH --mem=100gb
#SBATCH --array=0-1


#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


module purge

module load compiler/gnu/11.2
module load mpi/openmpi/4.1


# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate training_main

WORKING_DIR="/home/kit/stud/ukmwn/master_thesis/gpt-verite_/tools"
pushd $WORKING_DIR



LIST_DATASETS=("sc_loss" "no_sc_loss")
LIST_USE_LMASK=("--loss-mask" "")

DATASET=${LIST_DATASETS[$SLURM_ARRAY_TASK_ID]}
USE_LMASK=${LIST_USE_LMASK[$SLURM_ARRAY_TASK_ID]}

echo "DATASET: "$DATASET
echo "USE LOSS MASK: "$USE_LMASK


python preprocess_data_loss_mask.py \
            --input /pfs/work7/workspace/scratch/ukmwn-les_faits/les_faits_final/v1/${DATASET}.jsonl \
            --output-prefix /pfs/work7/workspace/scratch/ukmwn-les_faits/les_faits_final/v1/800T_pad/${DATASET} \
            --vocab ../../data/les_faits/tokenizer/gpt-ver-tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFGPTVerTokenizer \
            --loss-mask-multiple 1 \
            --pad-to-max-length \
            --discard-samples-smaller 800 \
            --append-eod \
            --workers 30 \
            $USE_LMASK




#             --pad-to-max-length \