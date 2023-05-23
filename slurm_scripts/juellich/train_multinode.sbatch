#!/bin/bash -x
#SBATCH --account=hai_verite
#SBATCH --job-name training-gpt_verite # Name for your job

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --time=00:30:00
#SBATCH --partition=booster

#SBATCH --mem=180
#SBATCH --gpus-per-node=4              # --gres=gpu:4


# job logic below

module --force purge
module load Stages/2023

module load GCC/11.3.0
module load NVHPC/23.1
module load OpenMPI/4.1.4

module load CUDA/11.7
module load git

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate training_main

WORKING_DIR="/p/project/hai_verite/training_repository/gpt-verite_"
pushd $WORKING_DIR

echo $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
scontrol show hostname $SLURM_JOB_NODELIST | sed 's/$/ slots=4/' > hostfile

GPUS_PER_NODE=4
NNODES=2
export SLURM_NTASKS=$(($GPUS_PER_NODE*$NNODES))

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000

python ./deepy.py ./train.py -d configs gpt-verite/1-4B_OPT.yml