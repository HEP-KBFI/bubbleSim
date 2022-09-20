#!/bin/bash

NUMJOBS=8

seq 1 $NUMJOBS | time singularity exec --nv --env LD_LIBRARY_PATH=/usr/local/cuda/lib64 ~/HEP-KBFI/singularity/base.simg parallel --gnu -j1 "build/bubbleSim.exe > log_gpu_par1_{#}.txt"
seq 1 $NUMJOBS | time singularity exec --nv --env LD_LIBRARY_PATH=/usr/local/cuda/lib64 ~/HEP-KBFI/singularity/base.simg parallel --gnu -j2 "build/bubbleSim.exe > log_gpu_par2_{#}.txt"
seq 1 $NUMJOBS | time singularity exec --nv --env LD_LIBRARY_PATH=/usr/local/cuda/lib64 ~/HEP-KBFI/singularity/base.simg parallel --gnu -j3 "build/bubbleSim.exe > log_gpu_par3_{#}.txt"
seq 1 $NUMJOBS | time singularity exec --nv --env LD_LIBRARY_PATH=/usr/local/cuda/lib64 ~/HEP-KBFI/singularity/base.simg parallel --gnu -j4 "build/bubbleSim.exe > log_gpu_par4_{#}.txt"
seq 1 $NUMJOBS | time singularity exec --nv --env LD_LIBRARY_PATH=/usr/local/cuda/lib64 ~/HEP-KBFI/singularity/base.simg parallel --gnu -j8 "build/bubbleSim.exe > log_gpu_par8_{#}.txt"

seq 1 $NUMJOBS | time singularity exec --env POCL_DEVICES=basic ~/HEP-KBFI/singularity/base.simg parallel --gnu -j1 "build/bubbleSim.exe > log_cpu_par1_{#}.txt"
seq 1 $NUMJOBS | time singularity exec --env POCL_DEVICES=basic ~/HEP-KBFI/singularity/base.simg parallel --gnu -j2 "build/bubbleSim.exe > log_cpu_par2_{#}.txt"
seq 1 $NUMJOBS | time singularity exec --env POCL_DEVICES=basic ~/HEP-KBFI/singularity/base.simg parallel --gnu -j3 "build/bubbleSim.exe > log_cpu_par3_{#}.txt"
seq 1 $NUMJOBS | time singularity exec --env POCL_DEVICES=basic ~/HEP-KBFI/singularity/base.simg parallel --gnu -j4 "build/bubbleSim.exe > log_cpu_par4_{#}.txt"
seq 1 $NUMJOBS | time singularity exec --env POCL_DEVICES=basic ~/HEP-KBFI/singularity/base.simg parallel --gnu -j8 "build/bubbleSim.exe > log_cpu_par8_{#}.txt"
