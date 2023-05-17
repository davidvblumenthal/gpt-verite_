#!/bin/bash
#SBATCH --job-name 160M_ET # Name for your job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00
#SBATCH --mem=180gb
#SBATCH --gpus-per-node=4              # --gres=gpu:4

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc



# check if this is the first or second submission
if [ -z "$SLURM_JOB_RESTART_COUNT"]; then
    count=1
else
    counts=$SLURM_JOB_RESTART_COUNT
fi

# job logic below

module purge

module load compiler/gnu/11.2
module load mpi/openmpi/4.1
module load devel/cuda/11.7

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate training_main

WORKING_DIR="/home/kit/stud/ukmwn/master_thesis/gpt-verite_"
pushd $WORKING_DIR

python ./deepy.py ./train.py -d configs gpt-verite/125M_padding_v1_ET.yml


# Submission logic for resubmit
if [ "$count" -lt 2 ]; then      # 2 submits the job 2 times - can be adjusted
    sbatch -p gpu_8 $0
fi