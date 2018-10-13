import numpy as np
import pytest

from ..autodh import create_dh_table
from ..joint import Joint
from .test_math_utils import sample_unit_vector


def sample_orthonormal_matrix(size=3):
    return np.linalg.qr(np.random.randn(size, size))[0]


def sample_transform_matrix():
    t = np.eye(4)
    t[:3, :3] = sample_orthonormal_matrix()
    t[:3, 3] = np.random.rand(3)
    return t


@pytest.mark.parametrize("repeat", range(10))
def test_identity_base_to_random_endeffector(repeat):
    base_frame = np.eye(4)
    ee_frame = sample_transform_matrix()
    joints = []
    dh_table = create_dh_table(joints, base_frame, ee_frame)
    assert np.allclose(dh_table.forward([]), ee_frame)


@pytest.mark.parametrize("repeat", range(10))
def test_random_base_to_random_endeffector(repeat):
    base_frame = sample_transform_matrix()
    ee_frame = sample_transform_matrix()
    joints = []
    dh_table = create_dh_table(joints, base_frame, ee_frame)
    base_to_ee = np.linalg.inv(base_frame).dot(ee_frame)
    assert np.allclose(dh_table.forward([]), base_to_ee)


@pytest.mark.parametrize("num_extra_joints", range(3))
@pytest.mark.parametrize("joint_type", [Joint.Type.Revolute, Joint.Type.Prismatic])
def test_identity_base_to_random_endeffector_with_random_joints(joint_type, num_extra_joints):
    base_frame = np.eye(4)
    ee_frame = sample_transform_matrix()
    joints = []
    joints.append(Joint([0, 0, 1], [0, 0, 0], joint_type))
    for _ in range(num_extra_joints):
        joints.append(Joint(sample_unit_vector(), np.random.rand(3), joint_type))
    joints.append(Joint(ee_frame[:3, 2], ee_frame[:3, 3], joint_type))
    dh_table = create_dh_table(joints, base_frame, ee_frame)
    assert np.allclose(dh_table.forward(np.zeros(num_extra_joints + 2)), ee_frame)
