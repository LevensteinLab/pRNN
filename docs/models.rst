Models
======

There are three types of models we focus on here. All are built upon the same class :class:`prnn.utils.Architectures.pRNN`, which initializes weights, chooses a base :class:`prnn.utils.thetaRNN.RNNCell` type, restructrures inputs, and runs a forward pass to generate predictions. The models below augment this base structure, by altering which agent observations and actions are visible when.

.. figure:: _static/FigureS1.png
    :alt: Overview of model architectures

    The three types of predictive networks supported by the package.

Next-step Prediction
--------------------

The Next-step prediction class :class:`prnn.utils.Architectures.NextStepRNN` is a special child of :class:`prnn.utils.Architectures.pRNN` with no masking actions or observations at future timesteps. 

.. autoclass:: prnn.utils.Architectures.NextStepRNN
    :members:
    :undoc-members:
    :show-inheritance:

At any given timestep ``t``, the pRNN will use the action and observation to predict what the egocentric view would be at timestep ``t+1``. The ``NextStepRNN`` class also supports prediction with no recurrent state (i.e. purely feed-forward).

Masked Prediction
-----------------

The Masked prediction class :class:`prnn.utils.Architectures.MaskedRNN` is another special child of :class:`prnn.utils.Architectures.pRNN` that allows variable mask lengths. 

.. autoclass:: prnn.utils.Architectures.MaskedRNN
    :members:
    :undoc-members:
    :show-inheritance:

Masks will hide observations taken at certain timesteps, so the agent must update and maintain its hidden state before passing another observation in as input. By default, the agent will have access to its actions, but there is an option to ``mask_actions`` as well. Actions can be offset backwards so that the agent uses the action made at ``t-1`` to inform the update at ``t``, rather than the action at ``t`` for this update (which moves the agent to the position at ``t+1``). 


Rollout-based Prediction
------------------------

The Rollout-based prediction class :class:`prnn.utils.Architectures.RolloutRNN` is an instantiation of :class:`prnn.utils.Architectures.pRNN_th` that does a ``k`` step "rollout" of predictions at each timestep ``t``. 

.. autoclass:: prnn.utils.Architectures.RolloutRNN
    :members:
    :undoc-members:
    :show-inheritance:

The number of rollout predictions per timestep is specified during construction. Rollout predictions are made in the context of the true actions made between timesteps ``1 ... T``. The strategy with how to use these actions during each rollout is also specified. The agent can either use the sequence of true actions during each rollout (i.e. the action at ``t+2`` is the same action made before the second prediction of the ``k`` step rollout), use only the first action, or hold the same action throughout the rollout. There is the option to carry over the hidden state after the rollout is finished to the next ``t`` timestep, or not.

Recall from :doc:`quickstart <quickstart.rst>` that we define a ``predictiveNet`` like the following:

.. code-block:: python

    #Make a pRNN
    num_neurons = 500
    pRNNtype = 'Masked'
    predictiveNet = PredictiveNet(env, hidden_size=num_neurons, pRNNtype=pRNNtype, useLN = True, inMask_length = 5)

Note that the ``pRNNtype`` is one of many predefined architectures (predefined in ``Architectures.py``) or can be one of the three generic types above. Recall that, for the ``Masked`` networks, ``h.shape`` will return ``[1, T, N]`` without batching and ``[1, T, N, B]`` with batching, where ``T`` is the number of timesteps, ``N`` is then number of neurons, and ``B`` is the batch_size. You can specify ``pRNNtype = TYPE`` where ``TYPE = {"NextStep", "Masked", "Rollout"}``, then pass in the required keyword arguments to further specify the network. For example:

.. code-block:: python

    #Make a pRNN
    num_neurons = 500
    pRNNtype = 'Rollout' 
    predictiveNet = PredictiveNet(env, hidden_size=num_neurons, pRNNtype=pRNNtype, use_ALN = False, k = 5, continuousTheta = False)

Also recall that, for ``Rollout`` architectures, ``h.shape`` will return ``[k, T, N]`` without batching and ``[k, T, N, B]`` with batching, where ``k`` is the number of rollout predictions per timestep. In addition to specifying various arguments for the architectures, you can also specify arguments to configure the initialization scheme (i.e. the ``init`` argument). For example:

.. code-block:: python

    #Make a pRNN
    num_neurons = 800
    pRNNtype = 'Rollout' 
    predictiveNet = PredictiveNet(env, hidden_size=num_neurons, pRNNtype=pRNNtype, use_ALN = False, k = 5, continuousTheta = False, init = "log_normal", sparsity = 0.05, eg_weight_decay=1e-8, eg_lr=2e-3, bias_lr=0.1)

This will initialize weights with values sampled from a log-normal distribution. Note that, if we would like to use log-normal initialization, we should specify a few extra parameters relating to the exponentiated gradient (EG) descent algorithm. It's a learning algorithm that preserves skewed (positive) log-normal weight distributions, sparse connectivity, and Dale's Law. This learning approach has yielded neurons that are more spatially-tuned. See the `related paper <https://www.biorxiv.org/content/10.1101/2024.10.25.620272v1>`__ for more details. The ``sparsity`` parameter handles the degree of this sparse connectivity. 

By default, however, the weights will instead be initialized with the Xavier/Glorot scheme (i.e. drawn from a scaled uniform distribution).