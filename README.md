# Predictive Replay

### Running scripts

If you want to run a python file somewhere in the repo structure, you will need to append the repo root to the pythonpath:

```
export PYTHONPATH=$PYTHONPATH:~/PredictiveReplay
```

(unless you have cloned the repo somewhere else than home).

For a notebook, before imports run:

```
import sys
repo_name = "PredictiveReplay"
if not sys.path[0].endswith(repo_name):# then [0] should be this nb's path
    sys.path.insert(0, sys.path[0][:sys.path[0].find(repo_name)+ len(repo_name)])
```


### Setup
To create virtual env called "PredictiveReplay" in home/mila/d/\<user>/venvs/  
Run: 

```. make_venv.sh```

Running the same command with the `load_venv.sh` script will activate this venv and load the required mila-modules. 
    
For vscode you need to add commands to load the mila-modules in your `.bash_profile`. See
https://mila-umontreal.slack.com/archives/C01TMEQ17J8/p1645743600721319?thread_ts=1645697530.184539&cid=C01TMEQ17J8

Following this advice my `.bashrc`, `.profile`, and `.bash_profile` are:

.bashrc
```   
# Bash interactive settings such as aliases, functions, completion, key bindings (that aren't in .inputrc)

[[ -s "$HOME/.profile" ]] && source "$HOME/.profile" # Load the default .profile
```
.profile
```
# Environment variables and other session settings

# since ssh'ing into a running job's node doesn't have the SLURM_TMPDIR
# variable set for some reason (TODO: look into it)
export SLURM_TMPDIR=/Tmp/slurm.$SLURM_JOB_ID.0

# for vs code debugging since the compute nodes can take a long time
# to spawn the debugpy debuggee
export DEBUGPY_PROCESS_SPAWN_TIMEOUT=500

# load the modules
module purge
module load python/3.7 
module load cuda/10.1/cudnn/7.6 
module load python/3.7/cuda/10.1/cudnn/7.6/pytorch/1.5.0
```
.bash_profile:
```
# [[ -s "$HOME/.profile" ]] && source "$HOME/.profile" # Load the default .profile
if [[ $- == *i* ]]; then . ~/.bashrc; fi # Load .bashrc if the shell is interactive
```

