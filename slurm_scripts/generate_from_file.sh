#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=20gb
#SBATCH --gpus-per-node=1            # --gres=gpu:4

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


python ./deepy.py ./generate.py -d configs 125M-ver-gen.yml text_gen_own.yml