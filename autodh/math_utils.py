# This Python file uses the following encoding: utf-8
import numpy as np


def skewed_lines(e1, k1, e2, k2):
    """Given two skew lines in Plücker coordinates, find the common perpendicular
    and compute the points p1, and p2 where the common perpendicular
    intersects with line 1 and line 2 respectively

    :param e1: direction vector of first line, as numpy array with shape (3,)
    :param k1: moment vector of first line, as numpy array with shape (3,)
    :param e2: direction vector of second line, as numpy array with shape (3,)
    :param k2: moment vector of second line, as numpy array with shape (3,)
    :returns: (common perpendicular, p1, p2), all three as numpy arrays with shape (3,)
    """
    n = np.cross(e1, e2)
    n /= np.linalg.norm(n)
    beta = np.arccos(e1.dot(e2))
    p1 = (k2.dot(n) - np.cos(beta) * k1.dot(n)) / np.sin(beta) * e1 + np.cross(e1, k1)
    p2 = (- k1.dot(n) + np.cos(beta) * k2.dot(n)) / np.sin(beta) * e2 + np.cross(e2, k2)
    return n, p1, p2


def intersecting_lines(e1, k1, e2, k2):
    """Given two intersecting lines in Plücker coordinates, find the common
    perpendicular and compute the point of intersection between line 1 and line 2

    :param e1: direction vector of first line, as numpy array with shape (3,)
    :param k1: moment vector of first line, as numpy array with shape (3,)
    :param e2: direction vector of second line, as numpy array with shape (3,)
    :param k2: moment vector of second line, as numpy array with shape (3,)
    :returns: (common perpendicular, p1, p2), all three as numpy arrays with shape (3,)
    """
    n = np.cross(e1, e2)
    n /= np.linalg.norm(n)
    if np.allclose(k1, 0) or np.allclose(k2, 0):
        origin_g = n
        e1_g = e1
        k1_g = k1 - np.cross(origin_g, e1)
        k2_g = k2 - np.cross(origin_g, e2)
        p_g = np.cross(k2_g, k1_g) / e1_g.dot(k2_g)
        p = origin_g + p_g
    else:
        p = np.cross(k2, k1) / e1.dot(k2)
    return n, p, p


def parallel_nonidentical_lines(c1, c2, e1, e2, k2):
    """Given two parallel, nonidentical, lines in Plücker coordinates,
    find a common perpendicular and compute the point where the common
    perpendicular intersects with line 2

    :param c1: point on first line, as numpy array with shape (3,)
    :param c2: point on second line, as numpy array with shape (3,)
    :param e1: direction vector of first line, as numpy array with shape (3,)
    :param e2: direction vector of second line, as numpy array with shape (3,)
    :param k2: moment vector of second line, as numpy array with shape (3,)
    :returns: (common perpendicular, p1, p2), all three as numpy arrays with shape (3,)
    """
    e1_p = np.cross(c2 - c1, e1)
    e1_pp = np.cross(e1, e1_p) / np.linalg.norm(e1_p)
    e1_temp, k1_temp = e1_pp, np.cross(c1, e1_pp)
    _, _, p2 = intersecting_lines(e1_temp, k1_temp, e2, k2)
    return e1_pp, c1, p2


def common_perpendicular_and_intersection_points(c1, e1, c2, e2, cp_hint):
    """Given two lines, compute their common perpendicular and its point of
    intersection, `p1` and `p2`, with line 1 and line 2 respectively

    Note, when the lines are identical a hint is required to break the tie
    between the infinitly many possibilities for the common perpendicular

    :param c1: point on first line, as numpy array with shape (3,)
    :param e1: direction of first line, as numpy array with shape (3,)
    :param c2: point on second line, as numpy array with shape (3,)
    :param e2: direction of second line, as numpy array with shape (3,)
    :param cp_hint: hint for common perpendicular, as numpy array with shape (3,)
    :returns: (common perpendicular, p1, p2), all three as numpy arrays with shape (3,)
    """
    assert np.isclose(np.linalg.norm(e1), 1) and np.isclose(np.linalg.norm(e2), 1), \
        "line direction vectors should be normalized"

    k1, k2 = np.cross(c1, e1), np.cross(c2, e2)  # s1, s2

    real = e1.dot(e2)
    dual = e1.dot(k2) + k1.dot(e2)
    if not np.isclose(dual, 0):
        cp, p1, p2 = skewed_lines(e1, k1, e2, k2)
    elif np.isclose(real, 1) or np.isclose(real, -1):
        cross_dual = np.cross(e1, k2) + np.cross(k1, e2)
        if not np.isclose(np.linalg.norm(cross_dual), 0):
            cp, p1, p2 = parallel_nonidentical_lines(c1, c2, e1, e2, k2)
        else:
            assert not np.allclose(cp_hint, 0), "hint can't be the zero vector"
            cp = cp_hint / np.linalg.norm(cp_hint)
            assert np.isclose(e1.dot(cp), 0), "hint must be perpendicular to the line"
            p1, p2 = c2, c2
    else:
        cp, p1, p2 = intersecting_lines(e1, k1, e2, k2)

    return cp, p1, p2
