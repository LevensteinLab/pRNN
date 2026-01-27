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

A Next-step pRNN can be defined similar to the Masked net in :doc:`quickstart <quickstart.rst>`:

.. code-block:: python

    #Make a pRNN
    num_neurons = 500
    pRNNtype = 'NextStep'
    predictiveNet = PredictiveNet(env, hidden_size=num_neurons, pRNNtype=pRNNtype)

At any given timestep ``t``, the pRNN will use the action and observation to predict what the egocentric view would be at timestep ``t+1``. Note that the ``NextStepRNN`` class also supports prediction with no recurrent state (i.e. purely feed-forward), with ``use_FF = False``.

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

The Rollout-based prediction class :class:`prnn.utils.Architectures.RolloutRNN` is an instantiation of :class:`prnn.utils.Architectures.pRNN_th` that does a ``k`` step "rollout" of predictions of future observations at each timestep ``t``. 

.. autoclass:: prnn.utils.Architectures.RolloutRNN
    :members:
    :undoc-members:
    :show-inheritance:

The number of rollout predictions per timestep is specified during construction. Rollout predictions are made in the context of the true actions made between timesteps ``1 ... T``. The strategy with how to use these actions during each rollout is also specified. The agent can either use the sequence of true actions during each rollout (i.e. the action at ``t+2`` is the same action made before the second prediction of the ``k`` step rollout), use only the first action, or hold the same action throughout the rollout. There is the option to carry over the hidden state after the rollout is finished to the next ``t`` timestep, or not (``continuousTheta = False``). To define a Rollout pRNN, you can use

.. code-block:: python

    #Make a pRNN
    num_neurons = 500
    pRNNtype = 'Rollout' 
    predictiveNet = PredictiveNet(env, hidden_size=num_neurons, pRNNtype=pRNNtype, k = 5, continuousTheta = False)

Let's take a look at the hidden unit activations in the rollout network:

.. code-block:: python    

    obs, act, _, _ = predictiveNet.collectObservationSequence(env,agent,10,discretize=True)
    obs_pred, obs_next, h  = predictiveNet.predict(obs,act, fullRNNstate=False)
    print(h.shape)

For the :class:`prnn.utils.Architectures.NextStepRNN` and :class:`prnn.utils.Architectures.MaskedRNN`, the shape of ``h`` will be ``[1, T, N]`` where ``T`` is the number of timesteps, and ``N`` is the number of neurons in the network. In the case of :class:`prnn.utils.Architectures.RolloutRNN`, the shape of ``h`` will be ``[k+1, T, N]`` where ``k`` is the number of rollout steps. 

Also recall that ``h.shape`` will return ``[k+1, T, N]`` without batching and ``[k+1, T, N, B]`` when trained from a dataset with batching, where ``k`` is the number of rollout predictions per timestep. 


Sparse-lognormal pRNN
---------------------

In addition to specifying various arguments for the architectures, you can also specify arguments to configure the initialization scheme (i.e. the ``init`` argument). Using sparse connectivity, with weights drawn from a lognormal distribution gives notably better place cells.

.. code-block:: python

    #Make a pRNN
    num_neurons = 800
    pRNNtype = 'Rollout' 
    predictiveNet = PredictiveNet(env, hidden_size=num_neurons, pRNNtype=pRNNtype, k = 5, init = "log_normal", sparsity = 0.05, eg_weight_decay=1e-8, eg_lr=2e-3, bias_lr=0.1)

This will initialize weights with values sampled from a log-normal distribution. Note that, if we would like to use log-normal initialization, we should specify a few extra parameters relating to the exponentiated gradient (EG) descent algorithm. It's a learning algorithm that preserves skewed (positive) log-normal weight distributions, sparse connectivity, and Dale's Law. See the `related paper <https://www.biorxiv.org/content/10.1101/2024.10.25.620272v1>`__ for more details. The ``sparsity`` parameter handles the degree of this sparse connectivity. 

By default, however, the weights will instead be initialized with the Xavier/Glorot scheme (i.e. drawn from a scaled uniform distribution).
