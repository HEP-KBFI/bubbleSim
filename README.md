### Execution platform

Run on CPU:

```
singularity exec --env POCL_DEVICES=basic ~/HEP-KBFI/singularity/base-new.simg build/bubbleSim.exe

...
platform: Portable Computing Language

...
========== Final results ==========
Count particles by mass and distance:
Count difference inside/outside: 0 / 0
Total particles energy: 1.50105e+06, Bubble energy: 431295, Total energy: 1.93234e+06
Energy/Initial Energy: 1

Time taken: 0 h 1 m 49 s 
```

Run on GPU
```
singularity exec --nv --env LD_LIBRARY_PATH=/usr/local/cuda/lib64 ~/HEP-KBFI/singularity/base-new.simg build/bubbleSim.exe
...
platform: NVIDIA CUDA
...
========== Final results ==========
Count particles by mass and distance:
Count difference inside/outside: 0 / 0
Total particles energy: 1.49872e+06, Bubble energy: 429790, Total energy: 1.92851e+06
Energy/Initial Energy: 1

Time taken: 0 h 1 m 6 s 
```
