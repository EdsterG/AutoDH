import numpy as np

from . import math_utils
from .dh_table import DHTable
from .joint import Joint


def _compute_row_in_standard_dh_table(z1, x1, o1, z2, x2, o2):
    """Given two frames, compute a row in the Denavit-Hartenberg table
    using the standard convention

    :param z1: z-axis of first frame, as numpy array with shape (3,)
    :param x1: x-axis of first frame, as numpy array with shape (3,)
    :param o1: origin of first frame, as numpy array with shape (3,)
    :param z2: z-axis of second frame, as numpy array with shape (3,)
    :param x2: x-axis of second frame, as numpy array with shape (3,)
    :param o2: origin of second frame, as numpy array with shape (3,)
    :returns: next row in dh table (a, alpha, d, theta)
    """
    alpha = np.arctan2(np.cross(z1, z2).dot(x2), z1.dot(z2))
    theta = np.arctan2(np.cross(x1, x2).dot(z1), x1.dot(x2))
    a = (o2 - o1).dot(x2)
    d = (o2 - o1).dot(z1)
    return d, theta, a, alpha


def _compute_row_in_modified_dh_table(z1, x1, o1, z2, x2, o2):
    """Given two frames, compute a row in the Denavit-Hartenberg table
    using the modified convention

    :param z1: z-axis of first frame, as numpy array with shape (3,)
    :param x1: x-axis of first frame, as numpy array with shape (3,)
    :param o1: origin of first frame, as numpy array with shape (3,)
    :param z2: z-axis of second frame, as numpy array with shape (3,)
    :param x2: x-axis of second frame, as numpy array with shape (3,)
    :param o2: origin of second frame, as numpy array with shape (3,)
    :returns: next row in dh table (a, alpha, d, theta)
    """
    alpha = np.arctan2(np.cross(z1, z2).dot(x1), z1.dot(z2))
    theta = np.arctan2(np.cross(x1, x2).dot(z2), x1.dot(x2))
    a = (o2 - o1).dot(x1)
    d = (o2 - o1).dot(z2)
    return d, theta, a, alpha


def _get_standard_dh_parameters(joints, base_frame, ee_frame):
    """Compute the standard Denavit-Hartenberg parameters

    :param joints: list of Joints
    :param base_frame: homogenous matrix representing the base frame, as numpy array with shape (4, 4)
    :param ee_frame: homogenous matrix representing the end-effector frame, as numpy array with shape (4, 4)
    :returns: (d, theta, a, alpha, joint_types), all as iterable objects
    """
    # Create a list of partial frames
    partial_frames = []
    partial_frames.append([base_frame[:3, 3], base_frame[:3, 0], base_frame[:3, 2], Joint.Type.Fixed])
    for j in joints:
        partial_frames.append([j.anchor, None, j.axis, j.type])
    partial_frames.append([ee_frame[:3, 3], None, ee_frame[:3, 2], Joint.Type.Fixed])
    partial_frames.append([ee_frame[:3, 3], ee_frame[:3, 0], ee_frame[:3, 2], Joint.Type.Fixed])

    # Determine x-axis for each partial frame
    frames = [partial_frames[0]]
    for i in range(1, len(partial_frames) - 1):
        o0, x0, z0, _ = frames[i - 1]
        o1, _, z1, jt = partial_frames[i]
        x1, _, o1_p = math_utils.common_perpendicular_and_intersection_points(o0, z0, o1, z1, cp_hint=x0)
        if np.allclose(x0.dot(x1), -1):
            x1 = -x1
        frames.append([o1_p, x1, z1, jt])
    frames.append(partial_frames[-1])

    # Create the DH table
    dh_table, joint_types = [], []
    for i in range(len(frames) - 1):
        o1, x1, z1, joint_type = frames[i]
        o2, x2, z2, _ = frames[i + 1]
        d, theta, a, alpha = _compute_row_in_standard_dh_table(z1, x1, o1, z2, x2, o2)
        if np.allclose([d, theta, a, alpha], 0) and joint_type == Joint.Type.Fixed:
            continue
        dh_table.append([d, theta, a, alpha])
        joint_types.append(joint_type)
    d, theta, a, alpha = np.transpose(dh_table)
    return d, theta, a, alpha, joint_types


def _get_modified_dh_parameters(joints, base_frame, ee_frame):
    """Compute the modified Denavit-Hartenberg parameters

    :param joints: list of Joints
    :param base_frame: homogenous matrix representing the base frame, as numpy array with shape (4, 4)
    :param ee_frame: homogenous matrix representing the end-effector frame, as numpy array with shape (4, 4)
    :returns: (d, theta, a, alpha, joint_types) all as iterable objects
    """
    # Create a list of partial frames
    partial_frames = []
    partial_frames.append([base_frame[:3, 3], base_frame[:3, 0], base_frame[:3, 2], Joint.Type.Fixed])
    partial_frames.append([base_frame[:3, 3], None, base_frame[:3, 2], Joint.Type.Fixed])
    for j in joints:
        partial_frames.append([j.anchor, None, j.axis, j.type])
    partial_frames.append([ee_frame[:3, 3], ee_frame[:3, 0], ee_frame[:3, 2], Joint.Type.Fixed])

    # Determine x-axis for each partial frame
    frames = [partial_frames[0]]
    for i in range(1, len(partial_frames) - 1):
        _, x0, _, _ = frames[i - 1]
        o1, _, z1, jt = partial_frames[i]
        o2, _, z2, _ = partial_frames[i + 1]
        x1, o1_p, _ = math_utils.common_perpendicular_and_intersection_points(o1, z1, o2, z2, cp_hint=x0)
        if np.allclose(x0.dot(x1), -1):
            x1 = -x1
        frames.append([o1_p, x1, z1, jt])
    frames.append(partial_frames[-1])

    # Create the DH table
    dh_table, joint_types = [], []
    for i in range(1, len(frames)):
        o0, x0, z0, _ = frames[i - 1]
        o1, x1, z1, joint_type = frames[i]
        d, theta, a, alpha = _compute_row_in_modified_dh_table(z0, x0, o0, z1, x1, o1)
        if np.allclose([d, theta, a, alpha], 0) and joint_type == Joint.Type.Fixed:
            continue
        dh_table.append([d, theta, a, alpha])
        joint_types.append(joint_type)
    d, theta, a, alpha = np.transpose(dh_table)
    return d, theta, a, alpha, joint_types


def get_dh_parameters(joints, base_frame, ee_frame, convention=DHTable.Convention.Standard):
    """Compute the Denavit-Hartenberg parameters

    :param joints: list of Joints
    :param base_frame: homogenous matrix representing the base frame, as numpy array with shape (4, 4)
    :param ee_frame: homogenous matrix representing the end-effector frame, as numpy array with shape (4, 4)
    :param convention: convention used to compute parameters, defaults to DHTable.Convention.Standard
    :returns: (d, theta, a, alpha, joint_types) all as iterable objects
    """
    if convention is DHTable.Convention.Modified:
        return _get_modified_dh_parameters(joints, base_frame, ee_frame)
    else:
        return _get_standard_dh_parameters(joints, base_frame, ee_frame)


def create_dh_table(joints, base_frame, ee_frame, convention=DHTable.Convention.Standard):
    d, theta, a, alpha, joint_types = get_dh_parameters(joints, base_frame, ee_frame, convention)
    return DHTable(d, theta, a, alpha, joint_types, convention)
