Dataloaders for pRNN Training
=============================

Here, we will discuss how to use the Dataloader. The environment is passed into the function that creates a DataLoader, and it automatically gets added to the environment. There are options to generate a trajectory (i.e. a sequence of actions & observations), or load from a provided path. Below is an example of this. First we import the tools to create a dataloader. 

.. code-block:: python

    import numpy as np
    import torch

    from prnn.utils.data import generate_trajectories, create_dataloader
    from prnn.utils.env import make_env
    from prnn.utils.agent import RandomActionAgent, RandomHDAgent
    from prnn.utils.predictiveNet import PredictiveNet

We also import ``generate_trajectories`` just to demonstrate how we can make a new dataset. Notably, ``create_dataloader`` will call this automatically. First, we need to instantiate an environment and agent (with corresponding action policy).

.. code-block:: python

    # Create the environment and the agent
    env = make_env(env_key='MiniGrid-LRoom-18x18-v0', package='gym-minigrid', act_enc='OneHotHD') #OneHotHD isn't working... 
    agent = RandomActionAgent(env.action_space, np.array([0.15,0.15,0.6,0.1,0,0,0]))


We can generate trajectories, or go straight to creating a dataloader. Ten thousand trajectories takes quite a while to generate. If you are running ``dataloader_example.ipynb`` yourself, consider shortening this to 100. 

.. code-block:: python

    # Generate trajectories if you want to do it as a separate step
    # (this can be skipped, as creating the dataloader will run it automatically)
    generate_trajectories(env, agent, n_trajs=10000, seq_length=500, folder='Data') 

    # Create the dataloader within the environment
    create_dataloader(env=env, agent=agent, n_trajs=10000, folder='Data', batch_size=1, seq_length=500, num_workers=0) 

Finally, we create a ``predictiveNet`` and train it.

.. code-block:: python
    
    predictiveNet = PredictiveNet(env, datal
    
    # Just to test if everything is working
    predictiveNet.trainingEpoch(env, agent,
                            sequence_duration=500,
                            num_trials=10)