# Multimodal Trajectory Prediction via Topological Invariance for Navigation at Uncontrolled Intersections
This is an implementation of CoRL 2020 paper "Multimodal Trajectory Prediction via Topological Invariance for Navigation at Uncontrolled Intersections" by Roh et al. [[arxiv](https://arxiv.org/abs/2011.03894)][[project](https://sites.google.com/view/multiple-topologies-prediction)]

## Citing the paper
If you use "Multimodal Trajectory Prediction via Topological Invariance for Navigation at Uncontrolled Intersections" in your research, please cite the paper:
```bibtex
@inproceedings{Roh2020Multimodal,
  title={Multimodal Trajectory Prediction via Topological Invariance for Navigation at Uncontrolled Intersections},
  author={Junha Roh and Christoforos Mavrogiannis and Rishabh Madan and Dieter Fox and Siddhartha S. Srinivasa},
  booktitle={Proceedings of the Conference on Robot Learning},
  year={2020},
}
```

## Instruction to run the code
For running the code, we have to install the prerequisites, setup an environment, run simulators and then run the code.
You would be able to run the evaluation code if you follow the instruction step by step.

***WARNING: one of scripts contains the code that modifies your `~/.bashrc` file. Please make a copy of your `~/.bashrc` file.***

### Install prerequisites and environment
We recommend using `anaconda` to setup the environment for the code.
Here's a list important libraries that are used in the code:
* python==3.8
* pytorch==1.5
These libraries will be installed if you follow the guide below.
 
#### Install guide
1. Install anaconda: either by manually or running `bash install_anaconda.sh`.
1. Setup an environment by running `conda env create -f env.yml`.
1. Activate the environment by running `conda activate mtp`.
1. With the conda environment `mtp` activated, run `pip install -r requirements.txt`.
1. With the conda environment `mtp` activated, run commands below.
```
pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-geometric
```
1. Run `bash run.sh` for download and extract CARLA binary, checkpoint and config files for the experiment.

### Run simulator with specific port
We have to load the map first (`Town04`) and specify the port to communicate with. 
Let us set `${ROOT}` a directory that you extract the code.
```
cd ${ROOT}/.carla && ./CarlaUE4.sh -benchmark -carla-port=4000 
```
Normally the simulator should run forever.
Then run a python code to load the map.
```python3 
cd ${ROOT}/.carla && python PythonAPI/util/config.py -m Town04 -p 4000
```  

### Run the evaluation code
Before running the code, we have to set the `$PYTHONPATH` to load the `CARLA` library by running `source env.sh`.
Then we will call `evaluator.py` with the experiment name.
Make sure there are `individual_trajs.pth`, `easy_*agent.json`, `hard_*agent.json` files in `${ROOT}`.
Following commands will run evaluations for easy scenarios in {2, 3, 4} agents cases.  
```
source env.sh
python evaluator.py -s easy_2agent.json -p 4000 --num_agent 2 --beta 8 --seed 1100 -a gn
python evaluator.py -s easy_3agent.json -p 4000 --num_agent 3 --beta 8 --seed 2000 -a gn
python evaluator.py -s easy_4agent.json -p 4000 --num_agent 4 --beta 8 --seed 4000 -a gn
```
If you replace `easy` with `hard` from the commands, you can also run hard scenarios.

## Structure of the code
* `agents`: overriding agent definitions from `carla`
* `controller`: implementation of 2D controller and cost functions
* `mtp`: implementation of the proposed, GNN-based model
* `msg`: contains some information for `ros`
* `utils`: contains utility functions

* `envlist.txt`: a list of packages installed at the time of release
* `evaluator.py`: a python script to run evaluation
