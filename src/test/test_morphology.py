'''
Unittests for the morphology pybind kernels.
'''
import sys
sys.path.append(sys.path[0]+"/../lib/cpp")
import cpu_seq.morphology as m_cpu_seq
import cpu.morphology as m_cpu
import gpu.morphology as m_gpu
import numpy as np
from scipy import ndimage as ndi
import pytest
from functools import partial
import datetime

# Parameters
implant_dims = 32
cross_width = 8
# TODO if implant_dims doesn't divide by radius, it doesn't work. Except for 2, which also fails.
rs = [4, 8, 16]
impls = [m_cpu_seq, m_cpu, m_gpu]
funcs = [('dilate', ndi.binary_dilation), ('erode', partial(ndi.binary_erosion, border_value=1))]

def sphere(n):
    xs = np.linspace(-1,1,n)
    return (xs[:,np.newaxis,np.newaxis]**2 + xs[np.newaxis,:,np.newaxis]**2 + xs[np.newaxis,np.newaxis,:]**2) <= 1

@pytest.mark.parametrize('r', rs)
@pytest.mark.parametrize('m', impls)
@pytest.mark.parametrize('op,nd', funcs)
def test_morphology(r, m, op, nd):
    implant_mask = np.random.randint(0, 2, (implant_dims, implant_dims, implant_dims), dtype=np.uint8)
    c = implant_dims // 2
    cross_start, cross_end = c - (cross_width // 2), c + (cross_width // 2)

    implant_mask[:,cross_start:cross_end,cross_start:cross_end] = True
    implant_mask[cross_start:cross_end,:,cross_start:cross_end] = True
    implant_mask[cross_start:cross_end,cross_start:cross_end,:] = True

    result = np.empty_like(implant_mask)
    f = getattr(m, f'{op}_3d_sphere')
    fsta = datetime.datetime.now()
    f(implant_mask, r, result)
    fend = datetime.datetime.now()

    vsta = datetime.datetime.now()
    verification = nd(implant_mask, sphere((2*r)+1))
    vend = datetime.datetime.now()

    assert np.allclose(verification, result)

    print (f'Testing the {m.__name__} implementation of {op}. Ran in {fend - fsta}, which is {(vend - vsta) / (fend - fsta)} times better than ndi')

if __name__ == '__main__':
    # TDOO move the data generation and ndi verification out to speed up running
    for r in rs:
        for m in impls:
            for op, nd in funcs:
                test_morphology(r, m, op, nd)
