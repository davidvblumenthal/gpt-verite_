#!/bin/bash
#SBATCH --job-name train_1-4B_v2
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --time=35:00:00
#SBATCH --mem=200gb
#SBATCH --gpus-per-node=4              # --gres=gpu:4

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


module purge    


module load compiler/gnu/11 # <-- HoreKa
module load mpi/openmpi/4.1
module load devel/cuda/11.6 # <-- HoreKa


# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate training_main

WORKING_DIR="/home/hk-project-test-lmcdm/ew9122/scratch_training/gpt-verite_"
pushd $WORKING_DIR



echo $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

scontrol show hostname $SLURM_JOB_NODELIST | sed 's/$/ slots=4/' > hostfile

GPUS_PER_NODE=4
NNODES=8
export SLURM_NTASKS=$(($GPUS_PER_NODE*$NNODES))

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000


python ./deepy.py ./train.py -d configs gpt-verite/optimal_models/1-4B_v2.yml