import numpy as np

from .joint import Joint


def _dh_row_to_matrix(a, alpha, d, theta):
    """Create matrix from row in DH table
    """
    mat = np.eye(4)
    mat[:3, 0] = [+np.cos(theta), +np.sin(theta), 0]
    mat[:3, 1] = [-np.sin(theta) * np.cos(alpha), +np.cos(theta) * np.cos(alpha), np.sin(alpha)]
    mat[:3, 2] = [+np.sin(theta) * np.sin(alpha), -np.cos(theta) * np.sin(alpha), np.cos(alpha)]
    mat[:3, 3] = [a * np.cos(theta), a * np.sin(theta), d]
    return mat


class DHTable:

    def __init__(self, d, theta, a, alpha, joint_types):
        self._dh = np.array([d, theta, a, alpha]).T
        self._jt = list(joint_types)
        self._num_dof = sum([x != Joint.Type.Fixed for x in self._jt])

    def forward(self, dof_values):
        assert len(dof_values) == self._num_dof
        i = 0
        mat = np.eye(4)
        for (d, theta, a, alpha), jt in zip(self._dh, self._jt):
            if jt is Joint.Type.Revolute:
                theta += dof_values[i]
                i += 1
            elif jt is Joint.Type.Prismatic:
                d += dof_values[i]
                i += 1
            mat = mat.dot(_dh_row_to_matrix(a, alpha, d, theta))
        return mat
