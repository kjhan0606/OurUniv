import numpy as np

from cf4_r2_sky_closed_split import close_group_marks, octant_from_xyz


def test_octant_and_graph_closure_do_not_move_count_points():
    xyz = np.array([[1, -1, 1], [-1, -1, -1], [1, -1, -1]])
    point_held = octant_from_xyz(xyz) == 5
    np.testing.assert_array_equal(point_held, [True, False, False])
    cf4, fp = close_group_marks(
        point_held, edge_point=[0, 0, 1], edge_cf4=[10, 11, 12],
        seed_cf4=[20], fp_cf4=[11, 12, 20, 21],
        fp_source=np.array(['A', 'A', 'B', 'B']),
        seed_fp=[False, False, False, False])
    assert cf4 == {10, 11, 12, 20, 21}
    assert fp == {'A', 'B'}
    np.testing.assert_array_equal(point_held, [True, False, False])


def test_invalid_edge_index_fails_closed():
    try:
        close_group_marks([False], [1], [7], [], [], np.array([]), [])
    except ValueError:
        pass
    else:
        raise AssertionError('out-of-range point index accepted')


def test_cross_method_anchor_is_a_group_graph_edge():
    # The first FP row links CF4 10 to source group A. The second edge is a
    # non-FP distance anchor, not an FP row, but must close CF4 20 as well.
    cf4, fp = close_group_marks(
        [True], [0], [10], [], [10,20],
        np.array(['A','A']), [False,False])
    assert cf4 == {10,20}
    assert fp == {'A'}
