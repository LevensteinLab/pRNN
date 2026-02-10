# Core predictive network
from .predictiveNet import (
    PredictiveNet,
    netOptions,
    lossOptions
)

# Network architectures
from .Architectures import (
    # Base classes
    pRNN,
    pRNN_th,
    pRNN_multimodal,
    NextStepRNN,
    MaskedRNN,
    RolloutRNN,
    # Autoencoder variants
    AutoencoderFF,
    AutoencoderRec,
    AutoencoderPred,
    AutoencoderFFPred,
    AutoencoderFF_LN,
    AutoencoderRec_LN,
    AutoencoderPred_LN,
    AutoencoderFFPred_LN,
    # Masked variants
    thRNN,
    thRNN_0win,
    thRNN_1win,
    thRNN_2win,
    thRNN_3win,
    thRNN_4win,
    thRNN_5win,
    thRNN_6win,
    thRNN_7win,
    thRNN_8win,
    thRNN_9win,
    thRNN_10win,
    thRNN_0win_prevAct,
    thRNN_1win_prevAct,
    thRNN_2win_prevAct,
    thRNN_3win_prevAct,
    thRNN_4win_prevAct,
    thRNN_5win_prevAct,
    thRNN_6win_prevAct,
    thRNN_7win_prevAct,
    thRNN_8win_prevAct,
    thRNN_9win_prevAct,
    thRNN_10win_prevAct,
    thRNN_1win_mask,
    thRNN_2win_mask,
    thRNN_3win_mask,
    thRNN_4win_mask,
    thRNN_5win_mask,
    # Rollout variants
    thcycRNN_3win,
    thcycRNN_4win,
    thcycRNN_5win,
    thcycRNN_5win_hold,
    thcycRNN_5win_full,
    thcycRNN_5win_first,
    thcycRNN_5win_holdc,
    thcycRNN_5win_fullc,
    thcycRNN_5win_firstc,
    thcycRNN_5win_hold_adapt,
    thcycRNN_5win_full_adapt,
    thcycRNN_5win_first_adapt,
    thcycRNN_5win_holdc_adapt,
    thcycRNN_5win_fullc_adapt,
    thcycRNN_5win_firstc_adapt,
    thcycRNN_5win_hold_prevAct,
    thcycRNN_5win_full_prevAct,
    thcycRNN_5win_first_prevAct,
    thcycRNN_5win_holdc_prevAct,
    thcycRNN_5win_fullc_prevAct,
    thcycRNN_5win_firstc_prevAct,
    # Log-normal init variants
    lognRNN_rollout,
    lognRNN_mask,
    # Multimodal variants
    multRNN_5win_i01_o01,
    multRNN_5win_i1_o0,
    multRNN_5win_i01_o0,
    multRNN_5win_i0_o1,
)

# CANN networks
from .CANNNet import (
    CANNnet,
    CANNRNN
)

# thetaRNN
from .thetaRNN import (
    LayerNormRNNCell,
    RNNCell,
)

# Action encodings
from .ActionEncodings import (
    OneHot,
    addHD,
    HDOnly,
    OneHotHD,
    OneHotHDPrevAct,
    SpeedHD,
    SpeedNextHD,
    Velocities,
    NoAct,
    Continuous,
    ContSpeedRotation,
    ContSpeedHD,
    ContSpeedOnehotHD,
    ContSpeedOnehotHDMiniworld,
)

# Agents
from .agent import (
    RandomActionAgent,
    RandomHDAgent,
    RatInABoxAgent,
    MiniworldRandomAgent,
    randActionSequence,
    create_agent
)

# Environment utilities
from .env import (
    make_env,
    make_farama_env,
    plot_env,
    RGBImgPartialObsWrapper_HD_Farama,
    wrappers
)

# Data utilities
from .data import (
    TrajDataset,
    TrajRawDataset,
    MergedTrajDataset,
    generate_trajectories,
    create_dataloader,
    mergeDatasets
)

# Checkpoint management
from .ckpts import (
    CkptKeys,
    save_pN,
    load_pN,
    load_pN_state_dict_only
)

# Linear decoder
from .LinearDecoder import (
    linearDecoder,
    linnet
)

# General utilities
from .general import (
    clumpyRandom,
    saveFig,
    savePkl,
    loadPkl,
    mkdir_p,
    fit_exp_linear,
    kl_divergence,
    delaydist,
    state2nap
)

from .enums import (
    pRNNtypes,
    MinigridEnvNames,
    ActionEncodingsEnum,
    AgentInputType,
    AgentType,
)

__all__ = [
    # Core
    'PredictiveNet',
    'netOptions',
    'lossOptions',
    # Base architectures
    'pRNN',
    'pRNN_th',
    'pRNN_multimodal',
    'NextStepRNN',
    'MaskedRNN',
    'RolloutRNN',
    # CANN
    'CANNnet',
    'CANNRNN',
    # Action encodings
    'OneHot',
    'addHD',
    'HDOnly',
    'OneHotHD',
    'OneHotHDPrevAct',
    'SpeedHD',
    'SpeedNextHD',
    'Velocities',
    'NoAct',
    'Continuous',
    'ContSpeedRotation',
    'ContSpeedHD',
    'ContSpeedOnehotHD',
    'ContSpeedOnehotHDMiniworld',
    # Agents
    'RandomActionAgent',
    'RandomHDAgent',
    'RatInABoxAgent',
    'MiniworldRandomAgent',
    'randActionSequence',
    'create_agent',
    # Environment
    'make_env',
    'make_farama_env',
    'plot_env',
    'RGBImgPartialObsWrapper_HD_Farama',
    'wrappers',
    # Data
    'TrajDataset',
    'TrajRawDataset',
    'MergedTrajDataset',
    'generate_trajectories',
    'create_dataloader',
    'mergeDatasets',
    # Checkpoints
    'CkptKeys',
    'save_pN',
    'load_pN',
    'load_pN_state_dict_only',
    # Decoder
    'linearDecoder',
    'linnet',
    # General utilities
    'clumpyRandom',
    'saveFig',
    'savePkl',
    'loadPkl',
    'mkdir_p',
    'fit_exp_linear',
    'kl_divergence',
    'delaydist',
    'state2nap',
    # Autoencoder variants
    'AutoencoderFF',
    'AutoencoderRec',
    'AutoencoderPred',
    'AutoencoderFFPred',
    'AutoencoderFF_LN',
    'AutoencoderRec_LN',
    'AutoencoderPred_LN',
    'AutoencoderFFPred_LN',
    # Masked variants
    'thRNN',
    'thRNN_0win',
    'thRNN_1win',
    'thRNN_2win',
    'thRNN_3win',
    'thRNN_4win',
    'thRNN_5win',
    'thRNN_6win',
    'thRNN_7win',
    'thRNN_8win',
    'thRNN_9win',
    'thRNN_10win',
    'thRNN_0win_prevAct',
    'thRNN_1win_prevAct',
    'thRNN_2win_prevAct',
    'thRNN_3win_prevAct',
    'thRNN_4win_prevAct',
    'thRNN_5win_prevAct',
    'thRNN_6win_prevAct',
    'thRNN_7win_prevAct',
    'thRNN_8win_prevAct',
    'thRNN_9win_prevAct',
    'thRNN_10win_prevAct',
    'thRNN_1win_mask',
    'thRNN_2win_mask',
    'thRNN_3win_mask',
    'thRNN_4win_mask',
    'thRNN_5win_mask',
    # Rollout variants
    'thcycRNN_3win',
    'thcycRNN_4win',
    'thcycRNN_5win',
    'thcycRNN_5win_hold',
    'thcycRNN_5win_full',
    'thcycRNN_5win_first',
    'thcycRNN_5win_holdc',
    'thcycRNN_5win_fullc',
    'thcycRNN_5win_firstc',
    'thcycRNN_5win_hold_adapt',
    'thcycRNN_5win_full_adapt',
    'thcycRNN_5win_first_adapt',
    'thcycRNN_5win_holdc_adapt',
    'thcycRNN_5win_fullc_adapt',
    'thcycRNN_5win_firstc_adapt',
    'thcycRNN_5win_hold_prevAct',
    'thcycRNN_5win_full_prevAct',
    'thcycRNN_5win_first_prevAct',
    'thcycRNN_5win_holdc_prevAct',
    'thcycRNN_5win_fullc_prevAct',
    'thcycRNN_5win_firstc_prevAct',
    # Log-normal init
    'lognRNN_rollout',
    'lognRNN_mask',
    # Multimodal
    'multRNN_5win_i01_o01',
    'multRNN_5win_i1_o0',
    'multRNN_5win_i01_o0',
    'multRNN_5win_i0_o1',
]