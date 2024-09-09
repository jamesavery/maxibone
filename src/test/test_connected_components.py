'''
Unit test for connected_components
'''
import cc3d
import lib.cpp.cpu.connected_components as cc
import datetime
from functools import partial
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as tp
import numpy as np
import scipy.ndimage as ndi

def test1():
    a = np.array([
        [0,0,0],
        [1,1,1],

        [1,0,2],
        [1,0,2],

        [1,0,2],
        [1,0,2],

        [1,1,1],
        [0,0,0],
    ], dtype=np.int64).reshape(4, 2, 1, 3)
    labels_a = np.array([1, 2, 2, 1], dtype=np.int64)
    print (a.shape, labels_a.shape)
    print (a.reshape(8,3))
    new_labels = cc.merge_labeled_chunks(a, labels_a, True)
    print (a.reshape(8,3))
    print (new_labels)

def test2():
    a = np.array([
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,0,2,2,2,0,3,0,0,5],
        [1,0,2,2,2,0,3,0,0,4,0,6],
        [0,0,0,0,0,0,0,0,0,0,0,0],
    ], dtype=np.int64).reshape(2, 2, 1, 12)
    labels_a = np.array([5,6], dtype=np.int64)
    print (a.shape, labels_a.shape)
    print (a.reshape(4,12))
    new_labels = cc.merge_labeled_chunks(a, labels_a, True)
    print (a.reshape(4,12))
    print (new_labels)

def test3():
    a = np.array([
        [0,0,0,0,0,0,0,0,0,4,0,0],
        [1,1,1,0,2,2,2,0,3,0,0,5],

        [1,0,2,0,3,0,4,0,0,5,0,6],
        [1,0,2,0,3,0,4,0,0,0,0,6],

        [1,0,2,0,3,0,4,0,0,0,0,6],
        [1,0,2,0,3,0,4,0,0,5,0,6],

        [1,0,2,2,2,0,3,0,4,0,0,6],
        [0,0,0,0,0,0,0,0,0,5,0,0],
    ], dtype=np.int64).reshape(4, 2, 1, 12)
    labels_a = np.array([5,6,6,6], dtype=np.int64)
    print (a.shape, labels_a.shape)
    print (a.reshape(8,12))
    new_labels = cc.merge_labeled_chunks(a, labels_a, True)
    print (a.reshape(8,12))
    print (new_labels)

def test4():
    a = np.array([
        [0,0,0,0,0,0,0,4,0,5,0,0],
        [0,0,2,2,2,2,2,0,0,5,0,0],
        [0,0,2,0,0,0,2,0,0,5,0,0],
        [1,0,2,0,3,0,2,0,0,5,0,6],

        [1,0,2,0,3,0,1,0,0,5,0,6],
        [1,0,0,0,0,0,1,0,0,5,0,0],
        [1,1,1,1,1,1,1,0,0,5,0,0],
        [0,0,0,0,0,0,0,4,0,5,0,0],
    ], dtype=np.int64).reshape(2, 4, 1, 12)
    labels_a = np.array([6,6], dtype=np.int64)

    print (a.shape, labels_a.shape)
    print (a.reshape(8,12))
    new_labels = cc.merge_labeled_chunks(a, labels_a, True)
    print (a.reshape(8,12))
    print (new_labels)

if __name__ == '__main__':
    test1()
    test2()
    test3()
    test4()