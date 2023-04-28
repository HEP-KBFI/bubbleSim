#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu=2G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/base.simg
singularity exec --nv \
	--env LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
	--env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
	$IMG \
	build/bubbleSim.exe bubbleSim/config.json
