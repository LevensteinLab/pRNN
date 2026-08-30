import torch
from enum import Enum
import os

from prnn.utils import PredictiveNet, pRNNtypes
from prnn.utils.Shell import FaramaMinigridShell


class CkptKeys(str, Enum):
    """String enums for checkpoint dictionary keys."""
    PRNN_TYPE = 'pRNNtype'
    PRNN_STATE_DICT = 'pRNN_state_dict'
    OPTIMIZER_STATE_DICT = 'optimizer_state_dict'
    HIDDEN_SIZE = 'hidden_size'
    OBS_SIZE = 'obs_size'
    ACT_SIZE = 'act_size'
    NUM_TRAINING_TRIALS = 'num_training_trials'
    NUM_TRAINING_EPOCHS = 'num_training_epochs'
    LEARNING_RATE = 'learning_rate'
    WEIGHT_DECAY = 'weight_decay'
    TRAIN_NOISE_MEAN_STD = 'train_noise_mean_std'
    ENCODER_STATE_DICT = 'encoder_state_dict'
    ENCODER_OPTIMIZER_STATE_DICT = 'encoder_optimizer_state_dict'


def save_pN(predictive_net: PredictiveNet, model_filepath: str):
    """
    Save PredictiveNet state dictionaries to specified model_filepath.
    """
    model_directory = os.path.dirname(model_filepath)
    os.makedirs(model_directory, exist_ok=True)
    
    state_dict = {
        CkptKeys.PRNN_TYPE: predictive_net.pRNNtype,
        CkptKeys.PRNN_STATE_DICT: predictive_net.pRNN.state_dict(),
        CkptKeys.OPTIMIZER_STATE_DICT: predictive_net.optimizer.state_dict(),
        CkptKeys.HIDDEN_SIZE: predictive_net.hidden_size,
        CkptKeys.OBS_SIZE: predictive_net.obs_size,
        CkptKeys.ACT_SIZE: predictive_net.act_size,
        CkptKeys.NUM_TRAINING_TRIALS: predictive_net.numTrainingTrials,
        CkptKeys.NUM_TRAINING_EPOCHS: predictive_net.numTrainingEpochs,
        CkptKeys.LEARNING_RATE: predictive_net.learningRate,
        CkptKeys.WEIGHT_DECAY: predictive_net.weight_decay,
        CkptKeys.TRAIN_NOISE_MEAN_STD: predictive_net.trainNoiseMeanStd,
    }
    
    # Save encoder if it exists and is trainable
    if hasattr(predictive_net.env_shell, 'encoder') and predictive_net.train_encoder:
        state_dict[CkptKeys.ENCODER_STATE_DICT] = predictive_net.env_shell.encoder.state_dict()
        if hasattr(predictive_net.env_shell.encoder, 'optimizer'):
            state_dict[CkptKeys.ENCODER_OPTIMIZER_STATE_DICT] = predictive_net.env_shell.encoder.optimizer.state_dict()
    
    torch.save(state_dict, model_filepath)


def load_pN(model_ckpt_filepath: str, 
            device: torch.device | str,
            pRNNtype : str, 
            env: FaramaMinigridShell | None = None, 
            predictive_net: PredictiveNet | None = None,) -> PredictiveNet:
    """
    Load PredictiveNet state dictionaries from model_filepath into an existing instance.
    """

    assert os.path.isfile(f"{model_ckpt_filepath}"), f"Network file {model_ckpt_filepath} does not exist."
    assert pRNNtype in pRNNtypes, f"pRNNtype {pRNNtype} is not a valid pRNNtype."

    # Normalize device to torch.device object
    device = torch.device(device)

    checkpoint = torch.load(model_ckpt_filepath, map_location=device, weights_only=False)

    if predictive_net is None:
        assert env is not None, "Environment must be provided if predictive_net is not."
        # Construct with the checkpoint's stored hidden_size (the fork version
        # ignored it and only worked when hidden_size happened to equal the
        # constructor default of 500).
        hidden_size = checkpoint.get(CkptKeys.HIDDEN_SIZE, 500)
        predictive_net = PredictiveNet(env=env, pRNNtype=pRNNtype, hidden_size=hidden_size)
    assert predictive_net.pRNNtype == checkpoint[CkptKeys.PRNN_TYPE], \
        f"Loading {checkpoint[CkptKeys.PRNN_TYPE]} into {predictive_net.pRNNtype} is not allowed."
    
    # Load main network and optimizer.
    #
    # PRE-BIAS CHECKPOINTS. `outlayer` gained a bias on 2026-08-30; a checkpoint
    # saved before that has no `outlayer.0.bias` key and `load_state_dict` is
    # strict, so it would raise `Missing key(s)`. The constructor zero-initialises
    # that bias, and a zero bias is mathematically the SAME FUNCTION as no bias -
    # so filling it is not a compatibility fudge, it is exact. Anything else
    # missing or unexpected still raises.
    state = checkpoint[CkptKeys.PRNN_STATE_DICT]
    missing, unexpected = predictive_net.pRNN.load_state_dict(state, strict=False)
    # BOTH names: the readout bias is registered twice, as `b_out` and as
    # `outlayer.0.bias`, exactly as `W_out` and `outlayer.0.weight` already are.
    # A pre-bias checkpoint is missing both.
    READOUT_BIAS = ("b_out", "outlayer.0.bias")
    pre_bias = [k for k in missing if k.endswith(READOUT_BIAS)]
    leftover = [k for k in missing if k not in pre_bias]
    if leftover or unexpected:
        raise RuntimeError(
            f"checkpoint does not match this architecture; missing {leftover}, "
            f"unexpected {list(unexpected)}"
        )
    if pre_bias:
        print(f"[load_pN] pre-bias checkpoint: {pre_bias} zero-filled "
              f"(identical to the bias-free network it was trained as)")
    predictive_net.pRNN.to(device)

    # PRE-BIAS OPTIMIZER STATE. `OutputBias` is APPENDED last, so a checkpoint
    # saved before it has one fewer param group and load_state_dict refuses on
    # the count. Appending a matching empty group restores the old groups at
    # their original indices and leaves the bias with fresh (zero) RMSprop
    # state - which is correct: a parameter that never trained has none.
    opt_state = checkpoint[CkptKeys.OPTIMIZER_STATE_DICT]
    live = predictive_net.optimizer.state_dict()
    if len(opt_state["param_groups"]) == len(live["param_groups"]) - 1:
        tail = live["param_groups"][-1]
        if tail.get("name") == "OutputBias":
            opt_state = dict(opt_state)
            opt_state["param_groups"] = list(opt_state["param_groups"]) + [tail]
            print("[load_pN] pre-bias optimizer state: appended an empty "
                  "OutputBias group")
    predictive_net.optimizer.load_state_dict(opt_state)
    
    # Move optimizer state tensors to match model device
    for state in predictive_net.optimizer.state.values():
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    # Load training statistics
    predictive_net.numTrainingTrials = checkpoint.get(CkptKeys.NUM_TRAINING_TRIALS, -1)
    predictive_net.numTrainingEpochs = checkpoint.get(CkptKeys.NUM_TRAINING_EPOCHS, -1)
    
    # Load encoder if present
    if CkptKeys.ENCODER_STATE_DICT in checkpoint and hasattr(predictive_net.env_shell, 'encoder'):
        predictive_net.env_shell.encoder.load_state_dict(checkpoint[CkptKeys.ENCODER_STATE_DICT]) #type: ignore
        
        if CkptKeys.ENCODER_OPTIMIZER_STATE_DICT in checkpoint and hasattr(predictive_net.env_shell.encoder, 'optimizer'): #type: ignore
            predictive_net.env_shell.encoder.optimizer.load_state_dict(checkpoint[CkptKeys.ENCODER_OPTIMIZER_STATE_DICT]) #type: ignore
    
    print(f"[load_pN] Completed loading. All tensors should now be on {device}")
    return predictive_net


def load_pN_state_dict_only(model_filepath: str, device: torch.device | str) -> dict:
    """
    Load only the pRNN state dictionary from the model directory.
    Useful when you only need the trained weights.
    """
    checkpoint = torch.load(model_filepath, map_location=device, weights_only=False)
    return checkpoint[CkptKeys.PRNN_STATE_DICT]
