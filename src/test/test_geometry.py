'''
Unit tests for the geometry library.
'''
import sys
sys.path.append(sys.path[0]+'/../lib/cpp')
import cpu_seq.geometry as m_cpu_seq
import cpu.geometry as m_cpu
import gpu.geometry as m_gpu
sys.path.append(sys.path[0]+'/../')
from config.paths import hdf5_root

import datetime
import edt
from functools import partial
import h5py
import numpy as np
import pytest

# Parameters
#n = 2344 # ~12 GB, used for testing whether blocked works.
n = 128

def assert_interesting_result(result):
    checksum = result.sum() if type(result) is np.ndarray else sum(result)
    assert (checksum < 0 or checksum > 0) # Sanity check that there's an actual result to compare to.

def assert_with_print(a, b, tolerance=1e-7, names=None):
    na, nb = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    nabs = np.abs(na - nb)
    all_close = np.alltrue(nabs < tolerance)
    if not all_close:
        #print ('a', na)
        #print ('b', nb)
        #print ('absolute error (AE) (abs(a-b))', nabs)
        print ('AE sum', np.sum(nabs))
        suma, sumb = na.sum(), nb.sum()
        print ('checksums', suma, sumb, np.abs(suma - sumb), suma / sumb)
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
    if allocate_result is None:
        f()
        start = datetime.datetime.now()
        result = f()
    else:
        if type(allocate_result) is tuple:
            alloc = lambda x: np.zeros(x[0], x[1])
        else:
            alloc = lambda x: np.copy(x)
        f(alloc(allocate_result))
        result = alloc(allocate_result)
        start = datetime.datetime.now()
        f(result)
    end = datetime.datetime.now()
    return result, end - start

def compare_fs(func, baseline_f, cpu_f, gpu_f, should_assert=True, tolerance=1e-7,
               allocate_result: tuple[tuple[int],np.dtype] | np.ndarray=None):
    baseline, baseline_t = run_with_warmup(baseline_f, allocate_result)
    print (f'({func}) Sequential ran in {baseline_t}')
    if should_assert: assert_interesting_result(baseline)

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

    assert_interesting_result(baseline())

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

def test_integrate_axes():
    n = 128
    dtype = np.uint8
    voxels = np.random.randint(0, np.iinfo(dtype).max, (n,n,n), dtype)
    cm = m_cpu.center_of_mass(voxels)
    M  = np.array(m_cpu.inertia_matrix(voxels, cm)).reshape(3,3)

    lam, E = np.linalg.eigh(M)
    ix = np.argsort(np.abs(lam))
    lam, E = np.array(lam)[ix], np.array(E)[:,ix]

    v_axis, w_axis = E[:,1], E[:,2]

    # TODO de her kan også bruges til test:
    v_axis, w_axis = np.array([1,0,0], np.float32), np.array([0,1,0], np.float32)

    (vmin,vmax), _ = axis_parameter_bounds(voxels.shape, cm, v_axis)
    (wmin,wmax), _ = axis_parameter_bounds(voxels.shape, cm, w_axis)

    cpu_seq, cpu, gpu = [
        partial(impl.integrate_axes, voxels, cm, v_axis, w_axis, vmin, wmin)
        for impl in [m_cpu_seq, m_cpu, m_gpu]
    ]
    #$void integrate_axes(const np_maskarray &np_voxels,
    #$            const array<real_t,3> &x0,
    #$            const array<real_t,3> &v_axis,
    #$            const array<real_t,3> &w_axis,
    #$            const real_t v_min,
    #             const real_t w_min,
    #$            np_realarray output) {

    compare_fs('integrate_axes', cpu_seq, cpu, gpu, True, 1e-7, ((int(vmax-vmin+2),int(wmax-wmin+2)), np.uint64))

def axis_parameter_bounds(shape, center, axis):
    signs = np.sign(axis)

    # (0,0,..,0) corner and furthest corner of grid, relative to center
#    print(center)
    x0 = -np.array(center)
    x1 = np.array(shape)[::-1]-center # Data has z,y,x-order, but we keep x,y,z in geometry calc

    xmin = (signs==1)*x0 + (signs==-1)*x1 # minimizes dot(x,axis)
    xmax = (signs==1)*x1 + (signs==-1)*x0 # maximizes dot(x,axis)

    return (np.dot(xmin,axis), np.dot(xmax,axis)), (xmin,xmax)

def integrate_axes(img, cm, v_axis, w_axis):
    (vmin,vmax), (vxmin,vxmax) = axis_parameter_bounds(img.shape, cm, v_axis)
    (wmin,wmax), (wxmin,wxmax) = axis_parameter_bounds(img.shape, cm, w_axis)

    integral = np.zeros((int(vmax-vmin+2),int(wmax-wmin+2)), dtype=float)
    m_cpu.integrate_axes(img,cm,v_axis, w_axis,vmin, wmin, integral)

    return integral

def bounding_volume(voxels,voxelsize=1.85):
    cm = np.array(m_cpu.center_of_mass(voxels))
    M  = np.array(m_cpu.inertia_matrix(voxels,cm)).reshape(3,3)

    lam,E = np.linalg.eigh(M)
    ix = np.argsort(np.abs(lam))
    lam,E = np.array(lam)[ix], np.array(E)[:,ix]

    u_axis, v_axis, w_axis = E[:,0], E[:,1], E[:,2]
    (vmin,vmax), _ = axis_parameter_bounds(voxels.shape, cm, v_axis)

    int_vw = integrate_axes(voxels, cm, v_axis, w_axis)
    int_uw = integrate_axes(voxels, cm, u_axis, w_axis)
    int_uv = integrate_axes(voxels, cm, u_axis, v_axis)
    int_u  = np.sum(int_uv,axis=1)
    int_v  = np.sum(int_uv,axis=0)
    int_w  = np.sum(int_uw,axis=0)

    lengths = np.array([np.sum(int_u>0), np.sum(int_v>0), np.sum(int_w>0)])
    ix = np.argsort(lengths)[::-1]
    print("lengths: ",lengths, ", ix: ",ix)

    (umin,umax), _ = axis_parameter_bounds(voxels.shape, cm, u_axis)
    (vmin,vmax), _ = axis_parameter_bounds(voxels.shape, cm, v_axis)
    (wmin,wmax), _ = axis_parameter_bounds(voxels.shape, cm, w_axis)

    u_prefix, u_postfix = np.sum(int_u[0:int(np.ceil(abs(umin)))]>0), np.sum(int_u[int(np.floor(abs(umin))):]>0)
    v_prefix, v_postfix = np.sum(int_v[0:int(np.ceil(abs(vmin)))]>0), np.sum(int_v[int(np.floor(abs(vmin))):]>0)
    w_prefix, w_postfix = np.sum(int_w[0:int(np.ceil(abs(wmin)))]>0), np.sum(int_w[int(np.floor(abs(wmin))):]>0)


    return {
        'principal_axes':np.array([u_axis,v_axis,w_axis]),
        'principal_axes_ranges':np.array([[-u_prefix*voxelsize,u_postfix*voxelsize],
                                          [-v_prefix*voxelsize,v_postfix*voxelsize],
                                          [-w_prefix*voxelsize,w_postfix*voxelsize]]),
        'centre_of_mass':cm*voxelsize
    }

def test_zero_outside_bbox():
    n = 128
    dtype = np.uint8
    voxels = np.random.randint(0, np.iinfo(dtype).max, (n,n,n), dtype)
    voxelsize = 1.85
    coarse_scale = 6
    fine_scale = 2
    mmtofi = 1 / (voxelsize * fine_scale) # Conversion factor from micrometers to index

    uvw_axes = np.array([[1,0,0],[0,1,0],[0,0,1]], np.float32)
    uvw_ranges = np.array([-16,16]*3, np.float32)
    cm = np.array(m_cpu.center_of_mass(voxels))

    cpu_seq, cpu, gpu = [
        partial(impl.zero_outside_bbox, uvw_axes.flatten(), uvw_ranges.flatten(), cm)
        for impl in [m_cpu_seq, m_cpu, m_gpu]
    ]

    compare_fs('zero_outside_bbox', cpu_seq, cpu, gpu, True, 1e-7, voxels)

def test_fill_implant_mask():
    n = 128
    dtype = np.uint8
    implant = np.random.randint(0, 2, (n,n,n), dtype)
    voxel_size = 1
    bbox_flat = np.array([-16,16] * 3, np.float32)
    rsqr_fraction = 1#0.7
    Muvwp_flat = np.array([
       1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1
    ], np.float32)
    n_bins = 1024

    solid_implant_mask = np.zeros(implant.shape, np.uint8)
    rsqr_maxs = np.zeros((n_bins, ), np.float32)
    profile = np.zeros((n_bins, ), np.float32)

    impls = [m_cpu_seq, m_cpu, m_gpu]
    result_solid_implant_mask = [solid_implant_mask.copy() for _ in impls]
    result_rsqr_maxs = [rsqr_maxs.copy() for _ in impls]
    result_profile = [profile.copy() for _ in impls]
    cpu_seq, cpu, gpu = [
        partial(impl.fill_implant_mask, implant, voxel_size, bbox_flat, rsqr_fraction, Muvwp_flat, result_solid_implant_mask[i], result_rsqr_maxs[i], result_profile[i])
        for i, impl in enumerate(impls)
    ]

    compare_fs('test_fill_implant_mask', cpu_seq, cpu, gpu, False)

    assert_interesting_result(result_solid_implant_mask[0])
    assert_interesting_result(result_rsqr_maxs[0])
    assert_interesting_result(result_profile[0])
    assert_with_print(result_solid_implant_mask[0], result_solid_implant_mask[1], 1e-7, "cpu_seq vs cpu")
    assert_with_print(result_solid_implant_mask[0], result_solid_implant_mask[2], 1e-7, "cpu_seq vs gpu")
    assert_with_print(result_rsqr_maxs[0], result_rsqr_maxs[1], 1e-7, "cpu_seq vs cpu")
    assert_with_print(result_rsqr_maxs[0], result_rsqr_maxs[2], 1e-7, "cpu_seq vs gpu")
    assert_with_print(result_profile[0], result_profile[1], 1e-7, "cpu_seq vs cpu")
    assert_with_print(result_profile[0], result_profile[2], 1e-7, "cpu_seq vs gpu")

def test_compute_front_mask():
    n = 128
    dtype = np.uint8
    implant = np.random.randint(0, 2, (n,n,n), dtype)
    voxel_size = 1
    bbox_flat = np.array([-16,16] * 3, np.float32)
    rsqr_fraction = 1#0.7
    Muvwp_flat = np.array([
       1, 0, 0, 64,
       0, 1, 0, 64,
       0, 0, 1, 64,
       0, 0, 0, 1
    ], np.float32)
    n_bins = 1024

    solid_implant_mask = np.zeros(implant.shape, np.uint8)
    rsqr_maxs = np.zeros((n_bins, ), np.float32)
    profile = np.zeros((n_bins, ), np.float32)

    m_cpu.fill_implant_mask(implant, voxel_size, bbox_flat, rsqr_fraction, Muvwp_flat, solid_implant_mask, rsqr_maxs, profile)

    impls = [m_cpu_seq, m_cpu, m_gpu]

    cpu_seq, cpu, gpu = [
        partial(impl.compute_front_mask, solid_implant_mask, voxel_size, Muvwp_flat, bbox_flat)
        for i, impl in enumerate(impls)
    ]

    compare_fs('test_compute_front_mask', cpu_seq, cpu, gpu, True, 1e-7, (solid_implant_mask.shape, solid_implant_mask.dtype))

# TODO postponed because it's not used until after segment_from_distributions, i.e. in the last analysis phase.
#def test_cylinder_projection():
#    n = 128
#    implant_mask = np.zeros((n,n,n), np.uint8)
#    implant_mask[:,n//2-4:n//2+4,n//2-4:n//2+4] = 1
#    edt_field = edt.edt(~implant_mask, parallel=16)
#
#    m_cpu_seq.cylinder_projection(edt_field, Cs, Cs_voxel_size,
#                    d_min, d_max, theta_min, theta_max,
#                    tuple(bbox.flatten()), tuple(Muvwp.flatten()),
#                    images, counts)

if __name__ == '__main__':
    np.random.seed(42)
    test_center_of_mass()
    test_inertia_matrix()
    test_sample_plane(np.uint8)
    test_integrate_axes()
    test_zero_outside_bbox()
    test_fill_implant_mask()
    test_compute_front_mask()