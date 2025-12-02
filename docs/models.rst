Models
======

There are three types of models we focus on here. All are built upon the same class :class:`prnn.utils.Architectures.pRNN`, which initializes weights, chooses a base :class:`prnn.utils.thetaRNN.RNNCell` type, restructrures inputs, and runs a forward pass to generate predictions. The models below augment this base structure, by altering which agent observations and actions are visible when.


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
    pRNNtype = 'thRNN_5win' #This will train a 5-step masked pRNN. 
    predictiveNet = PredictiveNet(env, hidden_size=num_neurons, pRNNtype=pRNNtype)

Note that the ``pRNNtype`` is one of many predefined architectures (predefined in ``Architectures.py``) or can be one of the three generic types above. You can specify ``pRNNtype = TYPE`` where ``TYPE = {"NextStep", "Masked", "Rollout"}``, then pass in the required keyword arguments to further specify the network. For example:

.. code-block:: python

    #Make a pRNN
    num_neurons = 500
    pRNNtype = 'Rollout' #This will train a 5-step masked pRNN. 
    predictiveNet = PredictiveNet(env, hidden_size=num_neurons, pRNNtype=pRNNtype, use_ALN = True, k = 10, continuousTheta = True)

