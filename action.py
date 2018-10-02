from enum import IntEnum

class Action(IntEnum):
    Forward, \
    TurnLeft, \
    TurnRight = range(3)
