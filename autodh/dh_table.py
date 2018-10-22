from enum import Enum

import numpy as np
from prettytable import PrettyTable

from .joint import Joint


def _standard_dh_row_to_matrix(a, alpha, d, theta):
    """Create matrix from row in DH table, standard convention
    """
    mat = np.eye(4)
    mat[:3, 0] = [+np.cos(theta), +np.sin(theta), 0]
    mat[:3, 1] = [-np.sin(theta) * np.cos(alpha), +np.cos(theta) * np.cos(alpha), np.sin(alpha)]
    mat[:3, 2] = [+np.sin(theta) * np.sin(alpha), -np.cos(theta) * np.sin(alpha), np.cos(alpha)]
    mat[:3, 3] = [a * np.cos(theta), a * np.sin(theta), d]
    return mat


def _modified_dh_row_to_matrix(a, alpha, d, theta):
    """Create matrix from row in DH table, modified convention
    """
    mat = np.eye(4)
    mat[:3, 0] = [+np.cos(theta), np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha)]
    mat[:3, 1] = [-np.sin(theta), np.cos(theta) * np.cos(alpha), np.cos(theta) * np.sin(alpha)]
    mat[:3, 2] = [0, -np.sin(alpha), np.cos(alpha)]
    mat[:3, 3] = [a, -d * np.sin(alpha), d * np.cos(alpha)]
    return mat


class DHTable:

    class Convention(Enum):
        Standard = 0
        Modified = 1

    def __init__(self, d, theta, a, alpha, joint_types, convention):
        self._dh = np.array([d, theta, a, alpha]).T
        self._jt = list(joint_types)
        self._convention = convention
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
            if self._convention is DHTable.Convention.Modified:
                mat = mat.dot(_modified_dh_row_to_matrix(a, alpha, d, theta))
            else:
                mat = mat.dot(_standard_dh_row_to_matrix(a, alpha, d, theta))
        return mat

    def __str__(self):
        table = PrettyTable()
        for i, field_name in enumerate(["d", "theta", "a", "alpha"]):
            if field_name in ['d', 'a']:
                column = self._dh[:, i] * 1000
            else:
                column = np.rad2deg(self._dh[:, i])
            table.add_column(field_name, column)
            table.float_format[field_name] = " 8.2"
        table.add_column("type", [jt.name for jt in self._jt])
        return str(table)
