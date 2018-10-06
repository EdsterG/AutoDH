from enum import Enum

import numpy as np


class Joint:

    class Type(Enum):
        Fixed = 0
        Revolute = 1
        Prismatic = 2

    def __init__(self, axis, anchor, joint_type):
        self._axis = np.array(axis, dtype=np.double)
        self._anchor = np.array(anchor, dtype=np.double)
        self._type = joint_type

    @property
    def axis(self):
        return self._axis.copy()

    @property
    def anchor(self):
        return self._anchor.copy()

    @property
    def type(self):
        return self._type
