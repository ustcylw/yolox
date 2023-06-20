import os, sys
import numpy as np
import torch
import operator as ops


CHECK_NONE = lambda data: data is None
CHECK_ELEM_NONE = lambda data: np.isnan(data).any()
CHECK_ELEM_ALL_NONE = lambda data: np.isnan(data).all()




CHECK_TYPE = lambda data, data_type: isinstance(data, data_type)
GET_TYPE = lambda data: type(data)

CHECK_TYPE_FLOAT = lambda data: isinstance(data, float)
CHECK_TYPE_INT = lambda data: isinstance(data, int)
CHECK_TYPE_STR = lambda data: isinstance(data, str)
CHECK_TYPE_NP_FLOAT = lambda data: isinstance(data, np.float32)
CHECK_TYPE_NP_INT = lambda data: isinstance(data, np.int32)
CHECK_TYPE_NP_ARRAY = lambda data: isinstance(data, np.ndarray)

def CHECK_NP_SHAPE(data, shape):
    '''
    '''
    data_ = data
    if not isinstance(data, np.ndarray):
        data_ = np.array(data)
    shape_ = data_.shape
    return ops.eq(list(shape_), list(shape))

def CHECK_NP_ELEM_EQ(data1, data2):
    data1_ = data1
    if not isinstance(data1, np.ndarray):
        data1_ = np.array(data1)
    data2_ = data2
    if not isinstance(data2, np.ndarray):
        data2_ = np.array(data2)
    return (data1_ == data2_).all()
