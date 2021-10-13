from typing import Tuple

import numpy as np

def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32") 
    res[0:int(shape[0]*1/3),0:int(shape[1]*1/2)] = .1
    res[0:int(shape[0]*1/3),int(shape[1]*1/2):] = .2
    res[int(shape[0]*1/3):shape[0], :int(shape[1]*1/2)] = .8
    res[int(shape[0]*1/3):int(shape[0]*2/3), :int(shape[1]*1/2)] = .2
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    res = np.zeros(shape=shape, dtype="float32")  # write your function instead of this one
    res[0:int(shape[0]*1/3),0:int(shape[1]*1/2)] = .2
    res[0:int(shape[0]*1/3),int(shape[1]*1/2):shape[1]] = .1
    res[int(shape[0]*1/3):shape[0], int(shape[1]*1/2):] = .8
    res[int(shape[0]*1/3):int(shape[0]*2/3), int(shape[1]*1/2):] = .2
    return res


