import numpy as np
import testing_fun as tf
from scipy.linalg import sqrtm


def get_lnz(beta=1):
    lnz = 0
    if beta < 0:
        operator = np.array([[1, np.exp(2 * beta)],
                             [np.exp(2 * beta), 1]])
        norm_ho = - beta + np.log(np.linalg.norm(operator))
        operator /= np.linalg.norm(operator)
    else:
        operator = np.array([[np.exp(- 2 * beta), 1],
                             [1, np.exp(- 2 * beta)]])
        norm_ho = beta + np.log(np.linalg.norm(operator))
        operator /= np.linalg.norm(operator)
    half_operator = sqrtm(operator)
    basic_tensor = ((tf.outer_parallel(
        half_operator, half_operator, half_operator)).sum(axis=0)).reshape(2, 2, 2)
    norm_b = np.log(np.linalg.norm(basic_tensor)) + 3 * norm_ho / 2
    basic_tensor /= np.linalg.norm(basic_tensor)
    pentagon = polygon_tensor_generator(basic_tensor=basic_tensor, n_side=5)
    norm_p = np.log(np.linalg.norm(pentagon)) + 5 * norm_b
    pentagon /= np.linalg.norm(pentagon)
    cell_tensor = polygon_tensor_generator(
        basic_tensor=pentagon, n_side=3)
    norm_c = np.log(np.linalg.norm(cell_tensor)) + 3 * norm_p
    cell_tensor /= np.linalg.norm(cell_tensor)
    whole_tensor = tf.tensor_contract(cell_tensor, cell_tensor, [[1, 2, 3], [3, 2, 1]])
    whole_tensor = tf.tensor_contract(
        whole_tensor, whole_tensor,
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [4, 9, 8, 7, 0, 5, 10, 3, 2, 1, 6, 11]])
    lnz += np.log(whole_tensor) + 4 * norm_c
    lnz = np.real(lnz)
    return lnz/60


def polygon_tensor_generator(basic_tensor, n_side=5):
    tmp = basic_tensor.copy()
    for ii in range(n_side-2):
        tmp = tf.tensor_contract(tmp, basic_tensor, [[-1], [0]])
    tmp = tf.tensor_contract(tmp, basic_tensor, [[-1, 0], [0, -1]])
    return tmp


