from enum import IntEnum

class Action(IntEnum):
    NoAction, \
    Accelerate, \
    Brake, \
    TurnLeft, \
    TurnRight, \
    AccelerateAndTurnLeft, \
    AccelerateAndTurnRight, \
    BrakeAndTurnLeft, \
    BrakeAndTurnRight = range(9)
