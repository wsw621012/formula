from enum import IntEnum

class Action(IntEnum):
    Forward = 1
    TurnLeft = 2
    TurnRight = 3

class Reverse(IntEnum):
    Backward = -1
    TurnLeft = -2
    TurnRight = -3
