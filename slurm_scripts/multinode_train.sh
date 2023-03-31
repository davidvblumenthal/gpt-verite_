#!/bin/bash
#SBATCH --job-name testing_multi_node_training # Name for your job
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --mem=50gb
#SBATCH --gpus-per-node=2              # --gres=gpu:4

#SBATCH --mail-user ukmwn@student.kit.edu     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc


module purge

module load compiler/gnu/11.2
module load mpi/openmpi/4.1
module load devel/cuda/11.7


# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate training_main

WORKING_DIR="/home/kit/stud/ukmwn/master_thesis/gpt-verite_"
pushd $WORKING_DIR



echo $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

scontrol show hostname $SLURM_JOB_NODELIST | sed 's/$/ slots=2/' > hostfile

GPUS_PER_NODE=2
NNODES=2
export SLURM_NTASKS=$(($GPUS_PER_NODE*$NNODES))

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000


python ./deepy.py ./train.py -d configs slurm_125M_single.yml

