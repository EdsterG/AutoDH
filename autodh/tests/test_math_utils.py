import numpy as np
import pytest

from .. import math_utils


def sample_unit_vector(size=3):
    """Randomly sample a unit vector

    :param size: size of the vector, defaults to 3
    :returns: unit vector, as numpy array with shape (size,)
    """
    e = np.zeros(size)
    while np.linalg.norm(e) < 1e-4:
        e = np.random.rand(size)
    return e / np.linalg.norm(e)


def sample_skewed_lines(force_zero_point=False):
    """Generate two random skewed lines

    :param force_zero_point: make the point on line 1 be zero, defaults to False
    :returns: (point on line 1, direction of line 1, point on line 2, direction of line 2)
    """
    e1 = sample_unit_vector()
    if force_zero_point:
        c1 = np.zeros(3)
    else:
        c1 = np.random.rand(3)

    e2 = sample_unit_vector()
    c2 = np.random.rand(3)

    return c1, e1, c2, e2


def sample_intersecting_lines(force_zero_point=False):
    """Generate two random intersecting lines

    :param force_zero_point: make the point on line 1 be zero, defaults to False
    :returns: (point on line 1, direction of line 1, point on line 2, direction of line 2)
    """
    e1 = sample_unit_vector()
    if force_zero_point:
        c1 = np.zeros(3)
    else:
        c1 = np.random.rand(3)

    e2 = sample_unit_vector()
    temp = c1 + e1 * (np.random.rand() * 2 - 1)
    c2 = temp + e2 * (np.random.rand() * 2 - 1)

    return c1, e1, c2, e2


def sample_parallel_lines(identical=False):
    """Generate two random parallel lines

    :param identical: make the two lines identical, defaults to False
    :returns: (point on line 1, direction of line 1, point on line 2, direction of line 2)
    """
    e1 = sample_unit_vector()
    c1 = np.random.rand(3)

    e2 = e1 * np.random.choice([1, -1])
    if identical:
        c2 = c1 + (np.random.rand() * 2 - 1) * e1
    else:
        c2 = np.random.rand(3)
        while np.allclose(c1, c2):
            c2 = np.random.rand(3)

    return c1, e1, c2, e2


def point_on_line(p, c, e):
    """Check if point `p` is on the line formed by point `c` and direction `e`

    Line `l` is described by:
        l = c + t*e for t in [-inf, inf]

    :param p: the point to check, as numpy array with shape (3,)
    :param c: a point on the line, as numpy array with shape (3,)
    :param e: direction of the line, as numpy array with shape (3,)
    :returns: True when p is on the line, False otherwise
    """
    if np.allclose(c, p):
        return True
    c2p = (p - c)
    c2p /= np.linalg.norm(c2p)
    return np.isclose(np.abs(e.dot(c2p)), 1)


def check_common_perpendicular_and_points(cp, p1, p2, c1, e1, c2, e2):
    assert np.isclose(np.linalg.norm(cp), 1), "common perpendicular must be normalized"
    assert np.allclose(e1.dot(cp), 0), "common perpendicular must be orthogonal to line 1"
    assert np.allclose(e2.dot(cp), 0), "common perpendicular must be orthogonal to line 2"
    assert point_on_line(p1, c1, e1), "p1 must be on line 1"
    assert point_on_line(p2, c2, e2), "p2 must be on line 2"

    if not np.allclose(p1, p2):
        m = p2 - p1
        m /= np.linalg.norm(m)
        assert np.allclose(cp, m)


@pytest.mark.parametrize("force_zero_point", [True, False])
@pytest.mark.parametrize("repeat", range(10))
def test_skewed_lines(force_zero_point, repeat):
    c1, e1, c2, e2 = sample_skewed_lines(force_zero_point)
    k1 = np.cross(c1, e1)
    k2 = np.cross(c2, e2)

    cp, p1, p2 = math_utils.skewed_lines(e1, k1, e2, k2)
    check_common_perpendicular_and_points(cp, p1, p2, c1, e1, c2, e2)


@pytest.mark.parametrize("force_zero_point", [True, False])
@pytest.mark.parametrize("repeat", range(10))
def test_intersecting_lines(force_zero_point, repeat):
    c1, e1, c2, e2 = sample_intersecting_lines(force_zero_point)
    k1 = np.cross(c1, e1)
    k2 = np.cross(c2, e2)

    cp, p1, p2 = math_utils.intersecting_lines(e1, k1, e2, k2)
    check_common_perpendicular_and_points(cp, p1, p2, c1, e1, c2, e2)

    assert np.allclose(p1, p2), "p1 and p2 must be equal"


@pytest.mark.parametrize("repeat", range(10))
def test_parallel_nonidentical_lines(repeat):
    c1, e1, c2, e2 = sample_parallel_lines(identical=False)
    k2 = np.cross(c2, e2)

    cp, p1, p2 = math_utils.parallel_nonidentical_lines(c1, c2, e1, e2, k2)
    check_common_perpendicular_and_points(cp, p1, p2, c1, e1, c2, e2)


def test_common_perpendicular_and_intersection_points(mocker):
    skewed_lines = mocker.patch(
        "autodh.math_utils.skewed_lines",
        return_value=[None, None, None]
    )
    intersecting_lines = mocker.patch(
        "autodh.math_utils.intersecting_lines",
        return_value=[None, None, None]
    )
    parallel_nonidentical_lines = mocker.patch(
        "autodh.math_utils.parallel_nonidentical_lines",
        return_value=[None, None, None]
    )

    c1, e1, c2, e2 = sample_skewed_lines()
    cp, p1, p2 = math_utils.common_perpendicular_and_intersection_points(c1, e1, c2, e2, None)
    assert skewed_lines.call_count == 1
    assert intersecting_lines.call_count == 0
    assert parallel_nonidentical_lines.call_count == 0
    assert not cp and not p1 and not p2

    c1, e1, c2, e2 = sample_intersecting_lines()
    cp, p1, p2 = math_utils.common_perpendicular_and_intersection_points(c1, e1, c2, e2, None)
    assert skewed_lines.call_count == 1
    assert intersecting_lines.call_count == 1
    assert parallel_nonidentical_lines.call_count == 0
    assert not cp and not p1 and not p2

    c1, e1, c2, e2 = sample_parallel_lines(identical=False)
    cp, p1, p2 = math_utils.common_perpendicular_and_intersection_points(c1, e1, c2, e2, None)
    assert skewed_lines.call_count == 1
    assert intersecting_lines.call_count == 1
    assert parallel_nonidentical_lines.call_count == 1
    assert not cp and not p1 and not p2

    c1, e1, c2, e2 = sample_parallel_lines(identical=True)
    cp_hint = np.cross(e1, [0, 0, 1])
    cp, p1, p2 = math_utils.common_perpendicular_and_intersection_points(c1, e1, c2, e2, cp_hint)
    assert skewed_lines.call_count == 1
    assert intersecting_lines.call_count == 1
    assert parallel_nonidentical_lines.call_count == 1
    check_common_perpendicular_and_points(cp, p1, p2, c1, e1, c2, e2)

    assert np.allclose(p1, p2), "p1 and p2 must be equal"


def test_intersecting_lines_edge_case():
    e1 = np.array([1., 0., 0.])
    e2 = np.array([0., 0., 1.])
    c1 = c2 = np.array([1., 0., 1.])
    k1 = np.cross(c1, e1)
    k2 = np.cross(c2, e2)
    cp, p1, p2 = math_utils.intersecting_lines(e1, k1, e2, k2)
    check_common_perpendicular_and_points(cp, p1, p2, c1, e1, c2, e2)

    assert np.allclose(p1, p2), "p1 and p2 must be equal"
