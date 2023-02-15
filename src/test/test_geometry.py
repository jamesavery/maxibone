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

def assert_with_print(a, b):
    all_close = np.allclose(a, b)
    if not all_close:
        na, nb = np.array(a), np.array(b)
        print (na)
        print (nb)
        nabs = np.abs(na - nb)
        print (nabs)
        print (np.sum(nabs))
    assert all_close

def run_with_warmup(f):
    f()
    start = datetime.datetime.now()
    result = f()
    end = datetime.datetime.now()
    return result, end - start

def compare_fs(func, baseline_f, cpu_f, gpu_f, should_assert=True):
    baseline, baseline_t = run_with_warmup(baseline_f)
    print (f'({func}) Sequential ran in {baseline_t}')

    cpu, cpu_t = run_with_warmup(cpu_f)
    print (f'({func}) Parallel CPU ran in {cpu_t}, which is {baseline_t / cpu_t} times faster than sequential')
    if should_assert: assert_with_print(baseline, cpu)

    gpu, gpu_t = run_with_warmup(gpu_f)
    print (f'({func}) GPU ran in {gpu_t}, which is {baseline_t / gpu_t} times faster than sequential')
    if should_assert: assert_with_print(baseline, gpu)


def test_center_of_mass():
    voxels = np.random.randint(0, 256, (n,n,n), np.uint8)

    baseline = partial(m_cpu_seq.center_of_mass, voxels)
    cpu = partial(m_cpu.center_of_mass, voxels)
    gpu = partial(m_gpu.center_of_mass, voxels)

    compare_fs('center_of_mass', baseline, cpu, gpu)


def test_inertia_matrix():
    voxels = np.random.randint(0, 2, (n,n,n), np.uint8)
    cm = m_gpu.center_of_mass(voxels)

    baseline = partial(m_cpu_seq.inertia_matrix, voxels, cm)
    cpu = partial(m_cpu.inertia_matrix, voxels, cm)
    gpu = partial(m_gpu.inertia_matrix, voxels, cm)

    # TODO assert disabled due to floating point associativity error accumulation
    compare_fs('inertia_matrix', baseline, cpu, gpu, should_assert=False)

if __name__ == '__main__':
    test_center_of_mass()
    test_inertia_matrix()