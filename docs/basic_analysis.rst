Basic Analysis with the pRNN Package
====================================

This tutorial explains the ``BasicAnalysis.py`` script, which demonstrates how to perform a comprehensive analysis of a trained pRNN model. The script loads a pre-trained network, calculates spatial representations, evaluates decoding performance, and generates visualizations of the network's cognitive map properties. We have provided a ``BasicAnalysis.ipynb`` notebook in the examples folder if you'd like to run this analysis interactively.

Imports and Setup
-----------------

The analysis script imports necessary utilities from the pRNN package, as well as standard data science libraries:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from prnn.utils.predictiveNet import PredictiveNet
    from prnn.utils.agent import RandomActionAgent
    import itertools
    import torch
    import random
    from prnn.utils.env import make_env
    from prnn.utils.general import saveFig
    from prnn.utils.figures import TrainingFigure
    import matplotlib.pyplot as plt
    from prnn.analysis.SpatialTuningAnalysis import SpatialTuningAnalysis
    from prnn.analysis.OfflineTrajectoryAnalysis import OfflineTrajectoryAnalysis
    from prnn.analysis.representationalGeometryAnalysis import representationalGeometryAnalysis

The key analysis classes are ``SpatialTuningAnalysis``, ``OfflineTrajectoryAnalysis``, and ``representationalGeometryAnalysis``. These perform different types of analyses on the trained network's hidden representations. We also set the folder where all figures will be saved:

.. code-block:: python

    savefolder = 'BasicAnalysisFigs'

Loading the Network and Environment
------------------------------------

First, we load a pre-trained pRNN model and retrieve the environment it was trained in:

.. code-block:: python

    #Example Net
    netname = 'Masked'
    netfolder = '/maskedk_panel/'
    exseed = 8
    predictiveNet = PredictiveNet.loadNet(netfolder+netname+'--s'+str(exseed))

The ``netname`` and ``netfolder`` variables specify which trained network to load. The ``exseed`` parameter identifies a specific training seed. The ``EnvLibrary`` attribute of the ``predictiveNet`` object contains the environments the network was trained on. We retrieve the first environment:

.. code-block:: python

    env = predictiveNet.EnvLibrary[0]

Setting Up the Agent
--------------------

Next, we define an agent that will interact with the environment during analysis. We use a ``RandomActionAgent`` with specific action probabilities:

.. code-block:: python

    agentname = 'RandomActionAgent'
    action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
    agent = RandomActionAgent(env.action_space,action_probability)

The ``action_probability`` array defines how likely each action is to be selected. In this case, forward movement (action 3) has a 60% probability, while rotations and other actions have lower probabilities. These probabilities should match the action space of your environment.

Calculating Spatial Representation
-----------------------------------

The core of the analysis is to extract the network's spatial representation. We train a linear decoder to map hidden unit activations to position:

.. code-block:: python

    place_fields, SI, decoder = predictiveNet.calculateSpatialRepresentation(env,agent,
                                                 trainDecoder=True, trainHDDecoder = True)

This function:

- Generates an agent trajectory through the environment
- Collects the network's hidden state activations at each position
- Trains a linear decoder to predict position from hidden states
- Returns ``place_fields`` (the spatial tuning of individual neurons), ``SI`` (spatial information), and the trained ``decoder`` object

Setting ``trainDecoder=True`` and ``trainHDDecoder=True`` trains decoders for both position and head direction.

Evaluating Decoding Performance
--------------------------------

Once the decoder is trained, we evaluate how well the network's representations can be decoded to recover position:

.. code-block:: python

    predictiveNet.calculateDecodingPerformance(env,agent,decoder,
                                                savename=netname, savefolder=savefolder,
                                              trajectoryWindow=5,
                                              timesteps=1000)

This function generates a trajectory and computes decoding error over time. The ``trajectoryWindow`` parameter sets the window for decoding, and ``timesteps`` specifies how long the trajectory is. Results are saved to ``savefolder`` with the filename based on ``savename``.

Spatial Tuning Analysis
-----------------------

The ``SpatialTuningAnalysis`` class examines individual neuron tuning curves and produces comparison figures:

.. code-block:: python

    STA = SpatialTuningAnalysis(predictiveNet,inputControl=True, untrainedControl=True)
    STA.TCExamplesFigure(netname,savefolder)

By setting ``inputControl=True`` and ``untrainedControl=True``, the analysis generates control comparisons. The ``TCExamplesFigure`` method creates a figure showing example tuning curves for neurons and saves it with the specified name to the save folder. This helps visualize which neurons develop spatial tuning and how selective they are.

Representational Geometry Analysis
-----------------------------------

The representational geometry analysis examines the structure of the network's representational space, particularly comparing activity during wake (active exploration) and sleep (spontaneous offline replay):

.. code-block:: python

    sleepnoise = 0.03
    isomap_neighbors = 15
    RGA = representationalGeometryAnalysis(predictiveNet, noisestd=sleepnoise,
                                           withIsomap=True, n_neighbors = isomap_neighbors)
    RGA.WakeSleepFigure(netname,savefolder)

The ``sleepnoise`` parameter controls the standard deviation of noise added to the network during offline analysis (simulating spontaneous activity). The ``isomap_neighbors`` parameter specifies how many neighbors are used in the Isomap dimensionality reduction. The ``WakeSleepFigure`` method generates a visualization comparing the structure of representations during wake and sleep states.

Offline Trajectory Analysis with Adaptation
--------------------------------------------

The ``OfflineTrajectoryAnalysis`` class analyzes spontaneous trajectories generated by the network during offline (sleep-like) activity. The first instantiation examines trajectories with synaptic adaptation:

.. code-block:: python

    b_adapt = 1
    tau_adapt = 100
    OTA_adapt = OfflineTrajectoryAnalysis(predictiveNet, noisestd=sleepnoise,
                                       withIsomap=False, decoder=decoder, 
                                          withAdapt=True, b_adapt = b_adapt, tau_adapt=tau_adapt,
                                          calculateViewSimilarity=True,
                                           compareWake=True)
    OTA_adapt.SpontTrajectoryFigure('adaptation',savefolder, trajRange=(150,250))

The ``b_adapt`` and ``tau_adapt`` parameters control the adaptation dynamics. Setting ``withAdapt=True`` enables adaptation mechanisms. The ``calculateViewSimilarity=True`` and ``compareWake=True`` flags enable additional analyses. The ``trajRange`` parameter specifies which portion of the trajectory to visualize.

Offline Trajectory Analysis with Action-Based Query
----------------------------------------------------

The second offline trajectory analysis examines trajectories generated when the network receives action queries:

.. code-block:: python

    OTA_query = OfflineTrajectoryAnalysis(predictiveNet, noisemag = 0, noisestd=sleepnoise,
                                   withIsomap=False, decoder=decoder,
                                         actionAgent=True, calculateViewSimilarity=True,
                                   compareWake=True)
    OTA_query.SpontTrajectoryFigure('actionquery',savefolder, trajRange=(110,150))

Here, ``actionAgent=True`` means the network receives action inputs during the offline trajectory. ``noisemag=0`` means no additional noise magnitude is applied. This analysis reveals whether the network can generate coherent trajectories that follow action-based commands.

Running the Analysis
--------------------

To run the basic analysis script, simply execute:

.. code-block:: bash

    python BasicAnalysis.py

All generated figures will be saved to the ``BasicAnalysisFigs/`` directory. The script will generate several comparison figures showing tuning curves, representational geometry, and offline replay trajectories. This provides a comprehensive view of the network's learned cognitive map and its ability to support both online perception and offline spatial reasoning.


