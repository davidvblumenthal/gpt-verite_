#!/bin/bash
#SBATCH --job-name training_125M-sc_loss # Name for your job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=200gb
#SBATCH --gpus-per-node=4              # --gres=gpu:4

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


python ./deepy.py ./train.py -d configs slurm_125M_single_sc_mask.yml

