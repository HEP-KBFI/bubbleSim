#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=2G
#SBATCH -o logs/slurm-%x-%j-%N.out

WORKDIR=/scratch/$USER/${SLURM_JOB_ID}
IMG=/home/software/singularity/base
BUBBLESIM_DIR=~/bubbleSim

#create a local dir on the worker node
mkdir -p $WORKDIR
cd $WORKDIR

#copy the kernel file to the local dir
cp $BUBBLESIM_DIR/bubbleSim/kernel.cl ./
cp $BUBBLESIM_DIR/build/bubbleSim.exe ./
ls -lrt .

singularity exec --nv \
	--env LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
	--env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        -B $WORKDIR -B $BUBBLESIM_DIR \
	$IMG \
	nice ./bubbleSim.exe $BUBBLESIM_DIR/bubbleSim/$1 &> log_1.txt &

singularity exec --nv \
	--env LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
	--env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        -B $WORKDIR -B $BUBBLESIM_DIR \
	$IMG \
	nice ./bubbleSim.exe $BUBBLESIM_DIR/bubbleSim/$2 &> log_2.txt &

singularity exec --nv \
	--env LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
	--env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        -B $WORKDIR -B $BUBBLESIM_DIR \
	$IMG \
	nice ./bubbleSim.exe $BUBBLESIM_DIR/bubbleSim/$3 &> log_3.txt &

nice +10 singularity exec --nv \
	--env LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
	--env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
        -B $WORKDIR -B $BUBBLESIM_DIR \
	$IMG \
	nice ./bubbleSim.exe $BUBBLESIM_DIR/bubbleSim/$4 &> log_4.txt &

#wait for parallel jobs to finish
wait

#print logs
cat log_*.txt

#remove temp dir
rm -Rf $WORKDIR
