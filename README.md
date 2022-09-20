### Instructions

Compile and run on the cluster:

```
export IMG=/home/software/singularity/base

singularity exec $IMG make -j8
cp bubbleSim/kernel.cl ./

#Run on CPU
singularity exec --env POCL_DEVICES=basic $IMG build/bubbleSim.exe bubbleSim/config.json

#Run on GPU (only on a GPU machine)
singularity exec --nv --env LD_LIBRARY_PATH=/usr/local/cuda/lib64 $IMG build/bubbleSim.exe bubbleSim/config.json
```
