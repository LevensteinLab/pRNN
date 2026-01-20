Dataloaders for pRNN Training
=============================

Here, we will discuss how to use the Dataloader. The Dataloader is a tool you can use to load a set of precomputed trajectories (i.e. sequences of observations and actions). This promises to save a significant amount of time because batching can be used at training time, and the agent doesn't need to interact with the environment at every time step. However, if your agent requires online learning and/or has a nonrandom policy, this will not work.

The environment is passed into the function that creates a DataLoader, and it automatically gets added to the environment. There are options to generate a trajectory (i.e. a sequence of actions & observations), or load from a provided path. Below is an example of this. First, we import the tools to create a dataloader. 

.. code-block:: python

    import numpy as np
    import torch

    from prnn.utils.data import generate_trajectories, create_dataloader
    from prnn.utils.env import make_env
    from prnn.utils.agent import RandomActionAgent, RandomHDAgent
    from prnn.utils.predictiveNet import PredictiveNet

We import ``generate_trajectories``. This will collecting sequences of observations and actions made by the agent traversing the environment. The dataset is stored with the environment object, so it can be conveniently used alongisde it during training. The ``create_dataloader`` function allows us to specify a folder with an existing dataset or generate one from scratch. If a path is specified, it checks the folder for the dataset and whether it has enough trajectories with long enough sequences. If one doesn't exist, it calls ``generate_trajectories`` and saves it. They can be used any time you use the same environment/agent combination, as long as you specify the same data path. 

First, we need to instantiate an environment and agent (with corresponding action policy).

.. code-block:: python

    # Create the environment and the agent
    env = make_env(env_key='LRoom-20x20-v0', package='farama-minigrid', act_enc='OneHotHD') #OneHotHD isn't working... 
    agent = RandomActionAgent(env.action_space, np.array([0.15,0.15,0.6,0.1,0,0,0]))


We can generate trajectories, or go straight to creating a dataloader. Ten thousand trajectories takes quite a while to generate. If you are running ``dataloader_example.ipynb`` yourself, consider shortening this to 100. 

.. code-block:: python

    # Generate trajectories if you want to do it as a separate step
    # (this can be skipped, as creating the dataloader will run it automatically)
    generate_trajectories(env, agent, n_trajs=10000, seq_length=500, folder='Data') 

    # Create the dataloader within the environment
    create_dataloader(env=env, agent=agent, n_trajs=10000, folder='Data', batch_size=1, seq_length=500, num_workers=0) 

Note that trajectories are saved directly to the path provided to the ``folder`` argument. This default saves it to a folder called ``Data`` in the current directory. If you're testing, you'll want to save the data to a location where you can store a lot of large files, such as scratch storage. There are cases where you may want to call ``generate_trajectories`` to precompute the dataset; ``create_dataloader`` should not be run at the same time for the same environment (e.g. across different jobs).

Finally, we create a ``predictiveNet`` and train it.

.. code-block:: python
    
    predictiveNet = PredictiveNet(env, dataloader=True)    
    
    # Just to test if everything is working
    predictiveNet.trainingEpoch(env, agent,
                            sequence_duration=500,
                            num_trials=10)

The dataloader can be toggled on and off in the initialization of ``PredictiveNet`` via the ``dataloader`` argument. Specifying this as true will pull observations and actions from the dataset for each epoch, via each call to ``predictiveNet.collectObservationSequence``.