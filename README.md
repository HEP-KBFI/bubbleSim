### Instructions

Compile and run on the cluster:

```
export IMG=/home/software/singularity/base.simg

singularity exec $IMG make clean
singularity exec $IMG make -j8

# Run with oclgrind

singularity exec $IMG oclgrind --data-races --uninitialized --log oclgrind.log ./build/bubbleSim.exe configs/test_oclgrind.json kernels/kernel.cl



#Run on CPU
singularity exec --env POCL_DEVICES=basic $IMG build/bubbleSim.exe bubbleSim/config.json kernels/kernel.cl

#Run on GPU (only on a GPU machine)
singularity exec --nv --env LD_LIBRARY_PATH=/usr/local/cuda/lib64 $IMG build/bubbleSim.exe bubbleSim/config.json kernels/kernel.cl
```

Submit to GPU queue:
```
#Run one simulation per GPU
sbatch scripts/bubblesim-gpu.sh

#Run 4 simulations in parallel on one GPU
sbatch scripts/bubblesim-gpu-p4.sh config1.json config2.json config3.json config4.json
```
