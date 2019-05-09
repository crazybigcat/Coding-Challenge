import numpy


def outer_parallel(a, *matrix):
    # need optimization
    for b in matrix:
        a = (a.repeat(b.shape[1], 1).reshape(a.shape + (-1,))
             * b.repeat(a.shape[1], 0).reshape(a.shape + (-1,))).reshape(a.shape[0], -1)
    return a


def tensor_contract(a, b, index):
    ndim_a = numpy.array(a.shape)
    ndim_b = numpy.array(b.shape)
    order_a = numpy.arange(len(ndim_a))
    order_b = numpy.arange(len(ndim_b))
    order_a_contract = numpy.array(order_a[index[0]]).flatten()
    order_b_contract = numpy.array(order_b[index[1]]).flatten()
    order_a_hold = numpy.setdiff1d(order_a, order_a_contract)
    order_b_hold = numpy.setdiff1d(order_b, order_b_contract)
    hold_shape_a = ndim_a[order_a_hold].flatten()
    hold_shape_b = ndim_b[order_b_hold].flatten()
    return numpy.dot(
        a.transpose(numpy.concatenate([order_a_hold, order_a_contract])).reshape(hold_shape_a.prod(), -1),
        b.transpose(numpy.concatenate([order_b_contract, order_b_hold])).reshape(-1, hold_shape_b.prod()))\
        .reshape(numpy.concatenate([hold_shape_a, hold_shape_b]))


def outer(a, *matrix):
    for b in matrix:
        a = numpy.outer(a, b).flatten()
    return a
