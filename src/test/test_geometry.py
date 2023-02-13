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

def run_with_warmup(f):
    f()
    start = datetime.datetime.now()
    result = f()
    end = datetime.datetime.now()
    return result, end - start

def test_center_of_mass():
    voxels = np.random.randint(0, 255, (n,n,n), np.uint8)

    baseline_f = partial(m_cpu_seq.center_of_mass, voxels)
    cpu_f = partial(m_cpu.center_of_mass, voxels)
    gpu_f = partial(m_gpu.center_of_mass, voxels)

    baseline, baseline_t = run_with_warmup(baseline_f)
    print (f'Sequential ran in {baseline_t}')

    cpu, cpu_t = run_with_warmup(cpu_f)
    print (f'Parallel CPU ran in {cpu_t}, which is {baseline_t / cpu_t} times faster than sequential')
    assert np.allclose(baseline, cpu)

    gpu, gpu_t = run_with_warmup(gpu_f)
    print (f'GPU ran in {gpu_t}, which is {baseline_t / gpu_t} times faster than sequential') 
    assert np.allclose(baseline, gpu)
    
if __name__ == '__main__':
    test_center_of_mass()