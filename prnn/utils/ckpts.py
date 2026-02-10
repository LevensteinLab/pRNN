import torch
from enum import Enum
import os

from prnn.utils.predictiveNet import PredictiveNet
from prnn.utils.Shell import FaramaMinigridShell
from prnn.utils.enums import pRNNtypes


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

    if predictive_net is None:
        assert env is not None, "Environment must be provided if predictive_net is not."
        predictive_net = PredictiveNet(env=env, pRNNtype=pRNNtype)
        
    checkpoint = torch.load(model_ckpt_filepath, map_location=device, weights_only=False)
    assert predictive_net.pRNNtype == checkpoint[CkptKeys.PRNN_TYPE], \
        f"Loading {checkpoint[CkptKeys.PRNN_TYPE]} into {predictive_net.pRNNtype} is not allowed."
    
    # Load main network and optimizer
    predictive_net.pRNN.load_state_dict(checkpoint[CkptKeys.PRNN_STATE_DICT])
    predictive_net.pRNN.to(device)

    predictive_net.optimizer.load_state_dict(checkpoint[CkptKeys.OPTIMIZER_STATE_DICT])
    
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
