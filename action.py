from enum import IntEnum

class Action(IntEnum):
    TurnRight = 0
    Forward = 1
    TurnLeft = 2

class Reverse(IntEnum):
    Backward = -1
    TurnLeft = -2
    TurnRight = -3
