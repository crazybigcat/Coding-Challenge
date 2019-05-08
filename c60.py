import numpy as np
import testing_fun as tf
from scipy.linalg import sqrtm


def get_lnz(beta=1):
    operator = np.array([[np.exp(- beta), np.exp(beta)],
                         [np.exp(beta), np.exp(- beta)]])
    half_operator = sqrtm(operator)
    basic_tensor = ((tf.outer_parallel(
        half_operator, half_operator, half_operator)).sum(axis=0)).reshape(2, 2, 2)
    basic_tensor = np.real(basic_tensor)
    norm_b = np.log(np.linalg.norm(basic_tensor))
    basic_tensor /= np.linalg.norm(basic_tensor)
    pentagon = polygon_tensor_generator(basic_tensor=basic_tensor, n_side=5)
    norm_p = np.log(np.linalg.norm(pentagon)) + 5 * norm_b
    pentagon /= np.linalg.norm(pentagon)
    cell_tensor = polygon_tensor_generator(
        basic_tensor=pentagon, n_side=3)
    norm_c = np.log(np.linalg.norm(cell_tensor)) + 3 * norm_p
    cell_tensor /= np.linalg.norm(cell_tensor)
    half_network = tf.tensor_contract(cell_tensor, cell_tensor, [[-1, 0, 1], [1, 0, -1]])
    norm_h = 2 * (np.log(np.linalg.norm(half_network)) + 2 * norm_c)
    # norm_h = 2 * np.log(np.linalg.norm(half_network))
    return norm_h/60


def polygon_tensor_generator(basic_tensor, n_side=5):
    tmp = basic_tensor.copy()
    for ii in range(n_side-2):
        tmp = tf.tensor_contract(tmp, basic_tensor, [[-1], [0]])
    tmp = tf.tensor_contract(tmp, basic_tensor, [[-1, 0], [0, -1]])
    return tmp


