'''
Unit tests for the geometry library.
'''
import sys
sys.path.append(sys.path[0]+'/../lib/cpp')
import cpu_seq.geometry as m_cpu_seq
import cpu.geometry as m_cpu
import gpu.geometry as m_gpu

import datetime
from functools import partial
import numpy as np
import pytest

# Parameters
#n = 2344 # ~12 GB, used for testing whether blocked works.
n = 128

def assert_with_print(a, b, tolerance=1e-7, names=None):
    na, nb = np.array(a), np.array(b)
    nabs = np.abs(na - nb)
    all_close = np.alltrue(nabs < tolerance)
    if not all_close:
        print ('a', na)
        print ('b', nb)
        print ('absolute error (AE) (abs(a-b))', nabs)
        print ('AE sum', np.sum(nabs))
        diffs = np.argwhere(nabs > tolerance)
        print (f'differing on {diffs.shape} elements')
        for i in diffs[:5]: # Only print 5 first
            print ('differing index (i), a[i], b[i] =', i, a[i[0], i[1]], b[i[0], i[1]])
        if not names is None:
            print (names)
    assert all_close

def run_with_warmup(f, allocate_result=None):
    '''
    Runs the given function and returns the result and how long time it took to run.

    @param allocate_result Defines whether the memory for the result should be allocated before running. If it should, it should be a tuple of the shape and the dtype of the array. None otherwise.
    '''
    alloc = lambda x: np.zeros(x[0], x[1])
    f() if allocate_result is None else f(alloc(allocate_result))
    if allocate_result is None:
        start = datetime.datetime.now()
        result = f()
    else:
        result = alloc(allocate_result)
        start = datetime.datetime.now()
        f(result)
    end = datetime.datetime.now()
    return result, end - start

def compare_fs(func, baseline_f, cpu_f, gpu_f, should_assert=True, tolerance=1e-7,
               allocate_result: tuple[tuple[int],np.dtype]=None):
    baseline, baseline_t = run_with_warmup(baseline_f, allocate_result)
    print (f'({func}) Sequential ran in {baseline_t}')

    cpu, cpu_t = run_with_warmup(cpu_f, allocate_result)
    print (f'({func}) Parallel CPU ran in {cpu_t}, which is {baseline_t / cpu_t:.02f} times faster than sequential')
    if should_assert: assert_with_print(baseline, cpu, tolerance, 'cpu_seq vs cpu')

    gpu, gpu_t = run_with_warmup(gpu_f, allocate_result)
    print (f'({func}) GPU ran in {gpu_t}, which is {baseline_t / gpu_t:.02f} times faster than sequential')
    if should_assert: assert_with_print(baseline, gpu, tolerance, 'cpu_seq vs gpu')

def test_center_of_mass():
    voxels = np.random.randint(0, 256, (n,n,n), np.uint8)

    baseline, cpu, gpu = [
        partial(impl.center_of_mass, voxels)
        for impl in [m_cpu_seq, m_cpu, m_gpu]
    ]

    compare_fs('center_of_mass', baseline, cpu, gpu, tolerance=1e-5)

def test_inertia_matrix():
    voxels = np.random.randint(0, 2, (n,n,n), np.uint8)
    cm = m_gpu.center_of_mass(voxels)

    baseline, cpu, gpu = [
        partial(impl.inertia_matrix, voxels, cm)
        for impl in [m_cpu_seq, m_cpu, m_gpu]
    ]

    # TODO assert disabled due to floating point associativity error accumulation
    compare_fs('inertia_matrix', baseline, cpu, gpu, should_assert=False)

@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_sample_plane(dtype):
    # TODO something that isn't just random data?
    n = 128
    voxels = np.random.randint(0, np.iinfo(dtype).max, (n,n,n), dtype)
    voxel_size = 42
    cm = m_cpu.center_of_mass(voxels)
    im = np.array(m_cpu.inertia_matrix(voxels, cm)).reshape((3,3))
    ls,E  = np.linalg.eigh(im)
    E[:,0] *= -1
    ix = np.argsort(np.abs(ls))
    ls, E = ls[ix], E[:,ix]
    UVW = E.T
    _, v_vec, w_vec = UVW
    cpu_seq, cpu, gpu = [
        partial(impl.sample_plane, voxels, voxel_size, cm, v_vec, w_vec, [0, 128, 0, 128])
        for impl in [m_cpu_seq, m_cpu, m_gpu]
    ]

    # TODO the function is unstable, even when they're all calling the sequential implementation, t least when comparing gcc against nvcc, but it differs at most with 1. Hence the higher tolerance for this test. Can be tested with something like for i in range(10000):
    compare_fs('sample_plane', cpu_seq, cpu, gpu, True, 1.1, ((64,64), np.float32))

if __name__ == '__main__':
    test_center_of_mass()
    test_inertia_matrix()
    test_sample_plane(np.uint8)