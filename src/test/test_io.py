'''
Unittests for the I/O pybind kernels.
'''
import sys
sys.path.append(sys.path[0]+"/../lib/cpp")
import cpu_seq.io as io_cpu_seq
import cpu.io as io_cpu
import gpu.io as io_gpu
import numpy as np
import tempfile
import os
import pytest

# TODO np.bool doesn't work. It works when writing, but numpy doesn't recognize that the memory has been updated. It works fine if data_read is a np.uint8 array, even though an np.bool array has been written.
dtypes_to_test = [np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64]
tmp_folder = tempfile._get_default_tempdir()
tmp_filename = next(tempfile._get_candidate_names())
tmp_file = f'{tmp_folder}/{tmp_filename}'
dim_size = 128
dim_shape = (dim_size, dim_size, dim_size)
partial_factor = 4
impls = [io_cpu_seq, io_cpu, io_gpu]

def random(shape, dtype):
    rnds = np.random.random(shape) * 100
    return rnds > .5 if dtype == bool else rnds.astype(dtype)

@pytest.mark.parametrize("impl", impls)
@pytest.mark.parametrize("dtype", dtypes_to_test)
def test_dtype(impl, dtype):
    individual_tmp_file = f'{tmp_file}.{dtype.__name__}'
    if os.path.exists(individual_tmp_file):
        os.remove(individual_tmp_file)
    data = random(dim_shape, dtype)
    partial = dim_size // partial_factor

    # Write out a new file
    impl.write_slice(data, individual_tmp_file, (0,0,0), dim_shape)
    assert os.path.getsize(individual_tmp_file) == data.nbytes

    # Read back and verify in chunks
    read_data = np.empty((partial, dim_size, dim_size), dtype=dtype)
    for i in range(partial_factor):
        impl.load_slice(read_data, individual_tmp_file, (i*partial,0,0), read_data.shape)
        assert np.allclose(data[i*partial:(i+1)*partial], read_data)

    # Append another layer
    data = np.append(data, random((partial, dim_size, dim_size), dtype), axis=0)
    impl.write_slice(data[dim_size:], individual_tmp_file, (dim_size,0,0), data.shape)
    assert os.path.getsize(individual_tmp_file) == data.nbytes

    # Read back and verify in chunks
    for i in range(partial_factor+1):
        impl.load_slice(read_data, individual_tmp_file, (i*partial,0,0), read_data.shape)
        assert np.allclose(data[i*partial:(i+1)*partial], read_data)

    # Overwrite one of the "middle" chunks
    data[partial:2*partial] = random((partial, dim_size, dim_size), dtype)
    impl.write_slice(data[partial:partial*2], individual_tmp_file, (partial,0,0), data.shape)
    assert os.path.getsize(individual_tmp_file) == data.nbytes

    # Read back and verify in chunks
    for i in range(partial_factor+1):
        impl.load_slice(read_data, individual_tmp_file, (i*partial,0,0), read_data.shape)
        assert np.allclose(data[i*partial:(i+1)*partial], read_data)

    # Write past where the file ends
    impl.write_slice(data, individual_tmp_file, (data.shape[0]*2,0,0), data.shape)
    assert os.path.getsize(individual_tmp_file) == 3 * data.nbytes

    # Check that the old data remains, the middle data is zeros, and that the new data is the same
    read_data = np.empty_like(data)
    for i in range(3):
        impl.load_slice(read_data, individual_tmp_file, (i*data.shape[0],0,0), data.shape)
        if i != 1:
            assert np.allclose(data, read_data)
        else:
            assert np.allclose(np.zeros_like(data), read_data)

    os.remove(individual_tmp_file)

if __name__ == '__main__':
    for impl in impls:
        for dtype in dtypes_to_test:
            print (f'Testing {impl.__name__} on {dtype.__name__}')
            test_dtype(impl, dtype)