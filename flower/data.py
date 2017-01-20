import itertools

import quantities as pq

from core.data import segment_volume


def cell_volume(src, dst):
    """

    :param src:
    :type src: flower.cell.Cell
    :param dst:
    :type dst: flower.cell.Cell
    :return:
    """

    segment_pairs = itertools.product(src.segments, dst.segments)
    total_volume = 0. * pq.bit
    for segment_pair in segment_pairs:
        total_volume += segment_volume(*segment_pair)

    return total_volume