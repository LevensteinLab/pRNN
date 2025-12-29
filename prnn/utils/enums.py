from enum import Enum, EnumMeta

class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True 

class MinigridEnvNames(str, Enum, metaclass=MetaEnum):
    LRoom = "MiniGrid-LRoom-v0"
    LRoomGoal = "MiniGrid-LRoom_Goal-v0"
    LRoomLineGreen = "MiniGrid-LRoom_LineGreen-v0"
    FourRooms = "MiniGrid-FourRooms-v0"
    FourRoomsObjs = "MiniGrid-FourRooms-Objects-v0"

class pRNNtypes(str, Enum, metaclass=MetaEnum):
    masked = "thRNN_5win"
    rollout = "thcycRNN_5win"
    masked_nextstep = "thRNN_5win_prevAct"

class ActionEncodingsEnum(str, Enum, metaclass=MetaEnum):
    OneHotHD = "OneHotHD"
    OneHotHDPrevAct = "OneHotHDPrevAct"
    OneHot = "OneHot"
    SpeedHD = "SpeedHD"
    SpeedNextHD = "SpeedNextHD"
    Velocities = "Velocities"
    NoAct = "NoAct"
    HDOnly = "HDOnly"
    Continuous = "Continuous"
    ContSpeedRotationRiaB = "ContSpeedRotationRiaB"
    ContSpeedHDRiaB = "ContSpeedHDRiaB"
    ContSpeedOneHotHDRiaB = "ContSpeedOneHotHDRiaB"
    ContSpeedOneHotHDMiniworld = "ContSpeedOneHotHDMiniworld"

class AgentInputType(str, Enum, metaclass=MetaEnum):
    H_PO = "pRNN+PO"
    Visual_FO = "Visual_FO"
    Visual_PO = "Visual_PO"
    PC = "PC"
    CANN = "CANN"
    PC_PO = "PC+PO"
    H = "pRNN"
    CANN_PO = "CANN+PO"
    CANN_norecurr = "CANN_norecurrence"

class AgentType(str, Enum, metaclass=MetaEnum):
    RANDOM = "random"
    AC = "curious"
