### Instructions

Compile and run on the cluster:

```
export IMG=/home/software/singularity/base

singularity exec $IMG make -j8
cp bubbleSim/config.json ./
cp bubbleSim/kernel.cl ./

#CPU
singularity exec --env POCL_DEVICES=basic $IMG build/bubbleSim.exe

#GPU
singularity exec --nv --env LD_LIBRARY_PATH=/usr/local/cuda/lib64 $IMG build/bubbleSim.exe
```
