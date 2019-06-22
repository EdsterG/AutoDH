import numpy as np
import pytest

from ..autodh import create_dh_table, get_dh_parameters
from ..dh_table import DHTable
from ..joint import Joint
from .test_math_utils import sample_unit_vector


def sample_orthonormal_matrix(size=3):
    return np.linalg.qr(np.random.randn(size, size))[0]


def sample_transform_matrix():
    t = np.eye(4)
    t[:3, :3] = sample_orthonormal_matrix()
    t[:3, 3] = np.random.rand(3)
    return t


def test_dh_to_str_doesnt_crash():
    str(DHTable([0.001], [0], [0.002], [np.pi], [Joint.Type.Fixed], DHTable.Convention.Standard))


@pytest.mark.parametrize("repeat", range(10))
@pytest.mark.parametrize("convention", DHTable.Convention)
def test_identity_base_to_random_endeffector(repeat, convention):
    base_frame = np.eye(4)
    ee_frame = sample_transform_matrix()
    joints = []
    dh_table = create_dh_table(joints, base_frame, ee_frame, convention)
    assert np.allclose(dh_table.forward([]), ee_frame)


@pytest.mark.parametrize("repeat", range(10))
@pytest.mark.parametrize("convention", DHTable.Convention)
def test_random_base_to_random_endeffector(repeat, convention):
    base_frame = sample_transform_matrix()
    ee_frame = sample_transform_matrix()
    joints = []
    dh_table = create_dh_table(joints, base_frame, ee_frame, convention)
    base_to_ee = np.linalg.inv(base_frame).dot(ee_frame)
    assert np.allclose(dh_table.forward([]), base_to_ee)


@pytest.mark.parametrize("num_extra_joints", range(3))
@pytest.mark.parametrize("joint_type", [Joint.Type.Revolute, Joint.Type.Prismatic])
@pytest.mark.parametrize("convention", DHTable.Convention)
def test_identity_base_to_random_endeffector_with_random_joints(joint_type, num_extra_joints, convention):
    base_frame = np.eye(4)
    ee_frame = sample_transform_matrix()
    joints = []
    joints.append(Joint([0, 0, 1], [0, 0, 0], joint_type))
    for _ in range(num_extra_joints):
        joints.append(Joint(sample_unit_vector(), np.random.rand(3), joint_type))
    joints.append(Joint(ee_frame[:3, 2], ee_frame[:3, 3], joint_type))
    dh_table = create_dh_table(joints, base_frame, ee_frame, convention)
    assert np.allclose(dh_table.forward(np.zeros(num_extra_joints + 2)), ee_frame)


@pytest.mark.parametrize("convention", DHTable.Convention)
def test_simple_3R_chain(convention):
    base_frame = np.eye(4)
    joints = [
        Joint([0, 0, 1], [0, 0, 0], Joint.Type.Revolute),
        Joint([0, -1, 0], [1, 0, 0], Joint.Type.Revolute),
        Joint([1, 0, 0], [1, 0, -2], Joint.Type.Revolute)
    ]
    ee_frame = np.eye(4)
    ee_frame[:3, :3] = [[+0, 0, 1],
                        [+0, 1, 0],
                        [-1, 0, 0]]
    ee_frame[:3, 3] = [1, 0, -2]

    d, th, a, al, joint_types = get_dh_parameters(joints, base_frame, ee_frame, convention)
    assert len(joint_types) == 3
    assert np.allclose([j.value for j in joint_types], [1, 1, 1])
    assert np.allclose(d, np.zeros(3))
    assert np.allclose(th, np.deg2rad([0, -90, 0]))
    if convention == DHTable.Convention.Modified:
        assert np.allclose(a, [0, 1, 2])
        assert np.allclose(al, np.deg2rad([0, 90, -90]))
    else:
        assert np.allclose(a, [1, 2, 0])
        assert np.allclose(al, np.deg2rad([90, -90, 0]))


@pytest.mark.parametrize("convention", DHTable.Convention)
def test_3R_chain_all_joints_inline(convention):
    base_frame = np.eye(4)
    joints = [
        Joint([0, 0, 1], [0, 0, 0], Joint.Type.Revolute),
        Joint([0, 0, -1], [0, 0, 1], Joint.Type.Revolute),
        Joint([0, 0, 1], [0, 0, 3], Joint.Type.Revolute)
    ]
    ee_frame = np.eye(4)
    ee_frame[:3, 3] = [0, 0, 3]

    d, th, a, al, joint_types = get_dh_parameters(joints, base_frame, ee_frame, convention)
    assert len(joint_types) == 3
    assert np.allclose([j.value for j in joint_types], [1, 1, 1])
    assert np.allclose(d, [1, -2, 0])
    assert np.allclose(th, np.zeros(3))
    assert np.allclose(a, np.zeros(3))
    if convention == DHTable.Convention.Modified:
        assert np.allclose(al, np.deg2rad([0, 180, 180]))
    else:
        assert np.allclose(al, np.deg2rad([180, 180, 0]))


@pytest.mark.parametrize("convention", DHTable.Convention)
def test_RRRP_chain(convention):
    base_frame = np.eye(4)
    joints = [
        Joint([0, 0, 1], [0, 0, 0], Joint.Type.Revolute),
        Joint([0, -1, 0], [0, 0, 0], Joint.Type.Revolute),
        Joint([0, -1, 0], [1, 0, 0], Joint.Type.Revolute),
        Joint([1, 0, 0], [1, 0, 0], Joint.Type.Prismatic)
    ]
    ee_frame = np.eye(4)
    ee_frame[:3, :3] = [[0, +0, 1],
                        [0, -1, 0],
                        [1, +0, 0]]
    ee_frame[:3, 3] = [1, 0, 0]

    d, th, a, al, joint_types = get_dh_parameters(joints, base_frame, ee_frame, convention)
    assert len(joint_types) == 4
    assert np.allclose([j.value for j in joint_types], [1, 1, 1, 2])
    assert np.allclose(d, np.zeros(4))
    assert np.allclose(th, np.deg2rad([0, 0, 90, 0]))
    if convention == DHTable.Convention.Modified:
        assert np.allclose(a, [0, 0, 1, 0])
        assert np.allclose(al, np.deg2rad([0, 90, 0, 90]))
    else:
        assert np.allclose(a, [0, 1, 0, 0])
        assert np.allclose(al, np.deg2rad([90, 0, 90, 0]))


@pytest.mark.parametrize("convention", DHTable.Convention)
def test_6R_chain(convention):
    base_frame = np.eye(4)
    joints = [
        Joint([0, 0, 1], [0, 0, 0], Joint.Type.Revolute),
        Joint([0, -1, 0], [0, 0, 0], Joint.Type.Revolute),
        Joint([0, -1, 0], [1, 0, 0], Joint.Type.Revolute),
        Joint([1, 0, 0], [3, 0, 0], Joint.Type.Revolute),
        Joint([0, -1, 0], [3, 0, 0], Joint.Type.Revolute),
        Joint([1, 0, 0], [3, 0, 0], Joint.Type.Revolute)
    ]
    ee_frame = np.eye(4)
    ee_frame[:3, :3] = [[0, +0, 1],
                        [0, -1, 0],
                        [1, +0, 0]]
    ee_frame[:3, 3] = [3, 0, 0]

    d, th, a, al, joint_types = get_dh_parameters(joints, base_frame, ee_frame, convention)
    assert len(joint_types) == 6
    assert np.allclose([j.value for j in joint_types], np.ones(6))
    assert np.allclose(d, [0, 0, 0, 2, 0, 0])
    assert np.allclose(th, np.deg2rad([0, 0, 90, 0, 0, 0]))
    if convention == DHTable.Convention.Modified:
        assert np.allclose(a, [0, 0, 1, 0, 0, 0])
        assert np.allclose(al, np.deg2rad([0, 90, 0, 90, -90, 90]))
    else:
        assert np.allclose(a, [0, 1, 0, 0, 0, 0])
        assert np.allclose(al, np.deg2rad([90, 0, 90, -90, 90, 0]))


@pytest.mark.parametrize("convention", DHTable.Convention)
def test_PUMA560_chain(convention):
    base_frame = np.eye(4)
    a2, a3 = 0.43, 0.02
    d3, d4 = 0.15, 0.43
    joints = [
        Joint([0, 0, 1], [0, 0, 0], Joint.Type.Revolute),
        Joint([0, 1, 0], [0, 0, 0], Joint.Type.Revolute),
        Joint([0, 1, 0], [a2, d3, 0], Joint.Type.Revolute),
        Joint([0, 0, -1], [a2 + a3, d3, -d4], Joint.Type.Revolute),
        Joint([0, 1, 0], [a2 + a3, d3, -d4], Joint.Type.Revolute),
        Joint([0, 0, -1], [a2 + a3, d3, -d4], Joint.Type.Revolute)
    ]
    ee_frame = np.eye(4)
    ee_frame[:3, :3] = [[1, +0, +0],
                        [0, -1, +0],
                        [0, +0, -1]]
    ee_frame[:3, 3] = [a2 + a3, d3, -d4]

    d, th, a, al, joint_types = get_dh_parameters(joints, base_frame, ee_frame, convention)
    assert len(joint_types) == 6
    assert np.allclose([j.value for j in joint_types], np.ones(6))
    assert np.allclose(d, [0, 0, d3, d4, 0, 0])
    assert np.allclose(th, np.zeros(6))
    if convention == DHTable.Convention.Modified:
        assert np.allclose(a, [0, 0, a2, a3, 0, 0])
        assert np.allclose(al, np.deg2rad([0, -90, 0, -90, 90, -90]))
    else:
        assert np.allclose(a, [0, a2, a3, 0, 0, 0])
        assert np.allclose(al, np.deg2rad([-90, 0, -90, 90, -90, 0]))
