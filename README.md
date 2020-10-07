# DiffTune: Optimizing CPU Simulator Parameters with Learned Differentiable Surrogates

DiffTune is a system for learning the parameters of x86 basic block CPU simulators from coarse-grained end-to-end measurements. 
Given a simulator, DiffTune learns its parameters by first replacing the original simulator with a differentiable surrogate, another function that approximates the original function; by making the surrogate differentiable, DiffTune is then able to apply gradient-based optimization techniques even when the original function is non-differentiable, such as is the case with CPU simulators. 
With this differentiable surrogate, DiffTune then applies gradient-based optimization to produce values of the simulator's parameters that minimize the simulator's error on a dataset of ground truth end-to-end performance measurements. Finally, the learned parameters are plugged back into the original simulator.

## Install
### Environment requirements
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
### Building
Inside of the source directory, run:
```
./build.sh
```

Then, to download the preprocessed BHive dataset and the released artifact, run: 
```
./download.sh
```

## Running DiffTune
To launch the docker container, inside of the source directory, run:
```
./run_docker.sh
```
Once inside the docker container (which you can exit with Ctrl-d or `exit`), run:
```
cd difftune
name=experiment-name
````
to go to the DiffTune code directory and set the experiment name (an arbitrary value that keeps data from different runs separated).

### Building the original dataset
To read the original dataset from BHive and the original parameters from llvm-mca, run the following:
```
python -m difftune.runner --name ${name} --task blocks
python -m difftune.runner --name ${name} --task default_params
```
These commands create pickle files in `data/${name}` with the basic block dataset and default parameter tables respectively.

Next, to generate the simulated dataset, run:
```
python -m difftune.runner --name ${name} --task sample_timings --sim mca --arch haswell --n-forks 100
```
This command samples parameter tables and runs them through llvm-mca on Haswell, writing the seed and result to `data/${name}/mca-haswell.csv`. 
`--n-forks=100` specifies to run 100 sampling workers in parallel, which should be tuned based on compute availability. 
This command does not ever terminate; when sufficient samples have been collected, just kill it with Ctrl-c, or manually truncate it to the desired length.

With the simulated dataset collected, to train the surrogate, run:
```
python -m difftune.runner --name ${name} --task approximation  --sim mca --arch haswell --model-name surrogate --device cuda:0 --epochs 6
```

Then to train the parameter table, run:
```
python -m difftune.runner --name ${name} --task parameters --sim mca --arch haswell --model-name surrogate --device cuda:0 --opt-alpha 0.05 --epochs 1
```

Finally, to extract the parameter table to a file (`data/${name}/surrogate-model-params-extracted`) and evaluate its test error / correlations, run:
```
python -m difftune.runner --name ${name} --task extract --sim mca --arch haswell --model-name surrogate
python -m difftune.runner --name ${name} --task validate --sim mca --arch haswell --model-name surrogate
```
