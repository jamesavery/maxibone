#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <inttypes.h>
#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <fstream>
using namespace std;
namespace py = pybind11;

typedef uint16_t voxel_type;
//typedef float    field_type;
typedef uint16_t field_type;
typedef uint8_t mask_type;
typedef float gauss_type;

#define INLINE __attribute__((always_inline)) inline

//#define VALID_VOXEL(voxel) (voxel != 0 && voxel >= vmin && voxel <= vmax) /* Voxel not masked, and within vmin,vmax range */
#define VALID_VOXEL(voxel) (voxel != 0) /* Voxel not masked */
#define GB_VOXEL ((1024 / sizeof(voxel_type)) * 1024 * 1024)

void gauss_filter_par_cpu(const py::array_t<mask_type> np_voxels,
                          const tuple<uint64_t, uint64_t, uint64_t> shape,
                          const py::array_t<gauss_type> np_kernel,
                          const uint64_t reps,
                          py::array_t<gauss_type> &np_result) {
    auto
        voxels_info = np_voxels.request(),
        kernel_info = np_kernel.request(),
        result_info = np_result.request();

    const mask_type      *voxels = static_cast<const mask_type*>(voxels_info.ptr);
    const gauss_type *kernel = static_cast<const gauss_type*>(kernel_info.ptr);

    gauss_type *result = static_cast<gauss_type*>(result_info.ptr);

    auto [Nz, Ny, Nx] = shape; // global shape TODO for blocked edition

    int64_t
        kernel_size = kernel_info.size,
        padding = kernel_size / 2, // kx should always be odd
        // Partial shape
        Pz = voxels_info.shape[0],
        Py = voxels_info.shape[1],
        Px = voxels_info.shape[2],
        // Result shape
        Rz = result_info.shape[0],
        Ry = result_info.shape[1],
        Rx = result_info.shape[2];

    assert(kernel_size % 2 == 1);

    gauss_type
        *tmp0 = (gauss_type *) calloc(Rz*Ry*Rx, sizeof(gauss_type)),
        *tmp1 = (gauss_type *) calloc(Rz*Ry*Rx, sizeof(gauss_type));

    #pragma omp parallel for
    for (int64_t i = 0; i < voxels_info.size; i++) {
        tmp0[i] = voxels[i] ? 1 : 0;
    }

    uint64_t iters = 3 * reps; // 1 pass for each dimension
    const int64_t strides[3] = {Py*Px,Px,1};
    const int64_t N[3] = {Pz,Py,Px};

    for (uint64_t rep = 0; rep < iters; rep++) {
        gauss_type *tin, *tout;
        if (rep % 2 == 1) {
            tin = tmp1;
            tout = tmp0;
        } else {
            tin = tmp0;
            tout = tmp1;
        }
        int64_t dim = rep % 3;

        #pragma omp parallel for
        for (int64_t z = 0; z < Pz; z++) {
            for (int64_t y = 0; y < Py; y++) {
                for (int64_t x = 0; x < Px; x++) {
                    int64_t output_index = z*strides[0] + y*strides[1] + x*strides[2];
                    auto mask_value = voxels[output_index];
                    if (dim % 3 == 2 && mask_value) {
                        tout[output_index] = 1;
                    } else {
                        int64_t 
                            X[3] = {z, y, x},
                            stride = strides[dim],
                            i_start = -min(padding, X[dim]),
                            i_end = min(padding, N[dim]-X[dim]-1);
                        gauss_type sum = 0;

                        #pragma omp simd reduction(+:sum)
                        for (int64_t i = i_start; i <= i_end; i++) {
                            int64_t voxel_index = output_index + stride*i;
                            sum += tin[voxel_index] * kernel[i+padding];
                        }

                        tout[output_index] = sum;
                    }
                }
            }
        }
    }

    memcpy(result, iters % 2 == 1 ? tmp1 : tmp0, Rz*Ry*Rx * sizeof(gauss_type));
    free(tmp0);
    free(tmp1);
}

pair<int,int> masked_minmax(const py::array_t<voxel_type> np_voxels) {
    // Extract NumPy array basearray-pointer and length
    auto voxels_info    = np_voxels.request();
    size_t image_length = voxels_info.size;
    const voxel_type *voxels = static_cast<const voxel_type*>(voxels_info.ptr);

    voxel_type voxel_min = max(voxels[0], voxel_type(1)), voxel_max = voxels[0];

    #pragma omp parallel for reduction(min:voxel_min) reduction(max:voxel_max)
    for (size_t i=0; i < image_length; i++) {
        voxel_min = min(voxel_min, voxels[i] > 0 ? voxels[i] : voxel_type(1));
        voxel_max = max(voxel_max, voxels[i]);
    }

    assert(voxel_min > 0);
    return make_pair(voxel_min,voxel_max);
}

pair<float,float> float_minmax(const py::array_t<float> np_field) {
    // Extract NumPy array basearray-pointer and length
    auto field_info    = np_field.request();
    size_t image_length = field_info.size;
    const float *field = static_cast<const float*>(field_info.ptr);

    float voxel_min = field[0], voxel_max = field[0];

    #pragma omp parallel for reduction(min:voxel_min) reduction(max:voxel_max)
    for (size_t i=0; i < image_length; i++) {
      float value = field[i];
      voxel_min = min(voxel_min, value);
      voxel_max = max(voxel_max, value);
    }

    return make_pair(voxel_min,voxel_max);
}

void load_slice(py::array_t<voxel_type> &np_data, string filename,
                const tuple<uint64_t, uint64_t, uint64_t> offset,
                const tuple<uint64_t, uint64_t, uint64_t> shape) {
    auto data_info = np_data.request();
    voxel_type *data = static_cast<voxel_type*>(data_info.ptr);
    ifstream file;
    file.open(filename.c_str(), ios::binary);
    auto [Nz, Ny, Nx] = shape;
    auto [oz, oy, ox] = offset;
    uint64_t flat_offset = (oz*Ny*Nx + oy*Nx + ox) * sizeof(voxel_type);
    file.seekg(flat_offset, ios::beg);
    file.read((char*) data, data_info.size * sizeof(voxel_type));
    file.close();
}

void write_slice(py::array_t<voxel_type> &np_data, uint64_t offset, string filename) {
    auto data_info = np_data.request();
    const voxel_type *data = static_cast<const voxel_type*>(data_info.ptr);
    ofstream file;
    file.open(filename.c_str(), ios::binary | ios::in);
    if (!file.is_open()) {
        file.clear();
        file.open(filename.c_str(), ios::binary);
    }
    file.seekp(offset * sizeof(voxel_type), ios::beg);
    file.write((char*) data, data_info.size * sizeof(voxel_type));
    file.close();
}

void append_slice(py::array_t<voxel_type> &np_data, string filename) {
    auto data_info = np_data.request();
    const voxel_type *data = static_cast<const voxel_type*>(data_info.ptr);
    ofstream file;
    file.open(filename.c_str(), ios::binary | ios::app);
    file.write((char*) data, data_info.size * sizeof(voxel_type));
    file.close();
}

// On entry, np_*_bins are assumed to be pre allocated and zeroed.
void axis_histogram_par_cpu(const py::array_t<voxel_type> np_voxels,
                            const tuple<uint64_t,uint64_t,uint64_t> offset,
                            const uint64_t block_size,
                            py::array_t<uint64_t> &np_x_bins,
                            py::array_t<uint64_t> &np_y_bins,
                            py::array_t<uint64_t> &np_z_bins,
                            py::array_t<uint64_t> &np_r_bins,
                            const tuple<uint64_t, uint64_t> center,
                            const tuple<double, double> vrange,
                            const bool verbose) {

    if (verbose) {
        auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
        tm local_tm = *localtime(&now);
        printf("Entered function at %02d:%02d:%02d\n", local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
    }

    py::buffer_info
        voxels_info = np_voxels.request(),
        x_info = np_x_bins.request(),
        y_info = np_y_bins.request(),
        z_info = np_z_bins.request(),
        r_info = np_r_bins.request();

    const uint64_t
        voxel_bins = x_info.shape[1],
        Nx = x_info.shape[0],
        Ny = y_info.shape[0],
        Nz = z_info.shape[0],
        Nr = r_info.shape[0];

    const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);

    uint64_t
        *x_bins = static_cast<uint64_t*>(x_info.ptr),
        *y_bins = static_cast<uint64_t*>(y_info.ptr),
        *z_bins = static_cast<uint64_t*>(z_info.ptr),
        *r_bins = static_cast<uint64_t*>(r_info.ptr);

    auto [z_start, y_start, x_start] = offset;
    uint64_t
        z_end   = min(z_start+block_size, Nz),
        y_end   = Ny,
        x_end   = Nx;

    auto [vmin, vmax] = vrange;
    auto [cy, cx] = center;

    if (verbose) {
        uint64_t memory_needed = omp_get_num_threads() * voxel_bins * sizeof(uint64_t);
        printf("\nStarting winning %p: (vmin,vmax) = (%g,%g), (Nx,Ny,Nz,Nr) = (%ld,%ld,%ld,%ld)\n",voxels,vmin, vmax, Nx,Ny,Nz,Nr);
        printf("Allocating working state memory (%ld bytes (%.02f Kbytes))\n", memory_needed, memory_needed/1024.);
        printf("Starting calculation\n");
        fflush(stdout);
    }

    auto start = chrono::steady_clock::now();

    #pragma omp parallel
    {
        uint64_t *tmp = (uint64_t*) calloc(voxel_bins, sizeof(uint64_t));

        // x_bins
        #pragma omp for nowait
        for (uint64_t x = x_start; x < x_end; x++) {
            // Init
            #pragma omp simd
            for (uint64_t i = 0; i < voxel_bins; i++)
                tmp[i] = 0;

            // Read & Compute
            #pragma omp simd collapse(2)
            for (uint64_t z = 0; z < z_end-z_start; z++) {
                for (uint64_t y = y_start; y < y_end; y++) {
                    uint64_t flat_idx = z*Ny*Nx + y*Nx + x;
                    auto voxel = voxels[flat_idx];
                    voxel = (voxel >= vmin && voxel <= vmax) ? voxel: 0; // Mask away voxels that are not in specified range
                    int64_t voxel_index = round(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                    if (voxel_index >= (int64_t) voxel_bins) {
                        fprintf(stderr,"Out-of-bounds error for index %ld: %ld > %ld:\n", flat_idx, voxel_index, voxel_bins);
                    } else if VALID_VOXEL(voxel) { // Voxel not masked, and within vmin,vmax range
                        tmp[voxel_index]++;
                    }
                }
            }

            // Store
            #pragma omp simd
            for (uint64_t i = 0; i < voxel_bins; i++)
                x_bins[x*voxel_bins + i] += tmp[i];
        }

        // y_bins
        #pragma omp for nowait
        for (uint64_t y = y_start; y < y_end; y++) {
            // Init
            #pragma omp simd
            for (uint64_t i = 0; i < voxel_bins; i++)
                tmp[i] = 0;

            // Read & Compute
            #pragma omp simd collapse(2)
            for (uint64_t z = 0; z < z_end-z_start; z++) {
                for (uint64_t x = x_start; x < x_end; x++) {
                    uint64_t flat_idx = z*Ny*Nx + y*Nx + x;
                    auto voxel = voxels[flat_idx];
                    voxel = (voxel >= vmin && voxel <= vmax) ? voxel: 0; // Mask away voxels that are not in specified range
                    int64_t voxel_index = round(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                    if (voxel_index >= (int64_t) voxel_bins) {
                        fprintf(stderr,"Out-of-bounds error for index %ld: %ld > %ld:\n", flat_idx, voxel_index, voxel_bins);
                    } else if VALID_VOXEL(voxel) { // Voxel not masked, and within vmin,vmax range
                        tmp[voxel_index]++;
                    }
                }
            }

            // Store
            #pragma omp simd
            for (uint64_t i = 0; i < voxel_bins; i++)
                y_bins[y*voxel_bins + i] += tmp[i];
        }

        // z_bins
        #pragma omp for nowait
        for (uint64_t z = 0; z < z_end-z_start; z++) {
            // Init
            #pragma omp simd
            for (uint64_t i = 0; i < voxel_bins; i++)
                tmp[i] = 0;

            // Read & Compute
            #pragma omp simd collapse(2)
            for (uint64_t y = y_start; y < y_end; y++) {
                for (uint64_t x = x_start; x < x_end; x++) {
                    uint64_t flat_idx = z*Ny*Nx + y*Nx + x;
                    auto voxel = voxels[flat_idx];
                    voxel = (voxel >= vmin && voxel <= vmax) ? voxel: 0; // Mask away voxels that are not in specified range
                    int64_t voxel_index = round(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                    if (voxel_index >= (int64_t) voxel_bins) {
                        fprintf(stderr,"Out-of-bounds error for index %ld: %ld > %ld:\n", flat_idx, voxel_index, voxel_bins);
                    } else if VALID_VOXEL(voxel) { // Voxel not masked, and within vmin,vmax range
                        tmp[voxel_index]++;
                    }
                }
            }

            // Store
            #pragma omp simd
            for (uint64_t i = 0; i < voxel_bins; i++)
                z_bins[(z+z_start)*voxel_bins + i] += tmp[i];
        }

        // r_bins
        #pragma omp for nowait collapse(2)
        for (uint64_t y = y_start; y < y_end; y++) {
            for (uint64_t x = x_start; x < x_end; x++) {
                // Init
                #pragma omp simd
                for (uint64_t i = 0; i < voxel_bins; i++)
                    tmp[i] = 0;

                uint64_t r = floor(sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy)));

                // Read and compute
                #pragma omp simd
                for (uint64_t z = 0; z < z_end-z_start; z++) {
                    uint64_t flat_idx = z*Ny*Nx + y*Nx + x;
                    auto voxel = voxels[flat_idx];
                    voxel = (voxel >= vmin && voxel <= vmax) ? voxel: 0; // Mask away voxels that are not in specified range
                    int64_t voxel_index = round(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                    if (voxel_index >= (int64_t) voxel_bins) {
                        fprintf(stderr,"Out-of-bounds error for index %ld: %ld > %ld:\n", flat_idx, voxel_index, voxel_bins);
                    } else if VALID_VOXEL(voxel) { // Voxel not masked, and within vmin,vmax range
                        tmp[voxel_index]++;
                    }
                }

                // Store
                for (uint64_t i = 0; i < voxel_bins; i++)
                    #pragma omp atomic
                    r_bins[r*voxel_bins + i] += tmp[i];
            }
        }

        #pragma omp barrier

        free(tmp);
    }

    auto end = chrono::steady_clock::now();

    if (verbose) {
        chrono::duration<double> diff = end - start;
        printf("Compute took %.04f seconds\n", diff.count());
        auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
        tm local_tm = *localtime(&now);
        printf("Exited function at %02d:%02d:%02d\n", local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
        fflush(stdout);
    }
}

void axis_histogram_par_gpu(const py::array_t<voxel_type> np_voxels,
                            const tuple<uint64_t,uint64_t,uint64_t> offset,
                            const uint64_t outside_block_size,
                            py::array_t<uint64_t> &np_x_bins,
                            py::array_t<uint64_t> &np_y_bins,
                            py::array_t<uint64_t> &np_z_bins,
                            py::array_t<uint64_t> &np_r_bins,
                            const tuple<uint64_t, uint64_t> center,
                            const tuple<double, double> vrange,
                            const bool verbose) {
#ifdef _OPENACC
    if (verbose) {
        auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
        tm local_tm = *localtime(&now);
        printf("Entered function at %02d:%02d:%02d\n", local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
    }

    py::buffer_info
        voxels_info = np_voxels.request(),
        x_info = np_x_bins.request(),
        y_info = np_y_bins.request(),
        z_info = np_z_bins.request(),
        r_info = np_r_bins.request();

    const uint64_t
        image_length = voxels_info.size,
        voxel_bins   = x_info.shape[1],
        Nx = x_info.shape[0],
        Ny = y_info.shape[0],
        Nz = z_info.shape[0],
        Nr = r_info.shape[0];

    voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);

    uint64_t memory_needed = ((Nx*voxel_bins)+(Ny*voxel_bins)+(Nz*voxel_bins)+(Nr*voxel_bins))*sizeof(uint64_t);
    uint64_t
        *x_bins = (uint64_t*)x_info.ptr,
        *y_bins = (uint64_t*)y_info.ptr,
        *z_bins = (uint64_t*)z_info.ptr,
        *r_bins = (uint64_t*)r_info.ptr;

    auto [z_start, y_start, x_start] = offset;
    uint64_t
        z_end   = min(z_start+outside_block_size, Nz),
        y_end   = Ny,
        x_end   = Nx;

    auto [vmin, vmax] = vrange;
    auto [cy, cx] = center;

    uint64_t block_size = 1 * GB_VOXEL;

    uint64_t iters = image_length / block_size;
    if (iters * block_size < image_length)
        iters++;

    uint64_t initial_block = min(image_length, block_size);

    if (verbose) {
        printf("\nStarting winning %p: (vmin,vmax) = (%g,%g), (Nx,Ny,Nz,Nr) = (%ld,%ld,%ld,%ld)\n",voxels,vmin, vmax, Nx,Ny,Nz,Nr);
        printf("Allocating result memory (%ld bytes (%.02f Mbytes))\n", memory_needed, memory_needed/1024./1024.);
        printf("Starting calculation\n");
        printf("Size of voxels is %ld bytes (%.02f Mbytes)\n", image_length * sizeof(voxel_type), (image_length * sizeof(voxel_type))/1024./1024.);
        printf("Blocksize is %ld bytes (%.02f Mbytes)\n", block_size * sizeof(voxel_type), (block_size * sizeof(voxel_type))/1024./1024.);
        printf("Doing %d blocks\n", iters);
        fflush(stdout);
    }

    auto start = chrono::steady_clock::now();

    // Copy the buffers to the GPU on entry and back to host on exit
    #pragma acc data copy(x_bins[:Nx*voxel_bins], y_bins[:Ny*voxel_bins], z_bins[:Nz*voxel_bins], r_bins[:Nr*voxel_bins])
    {
        // For each block
        for (uint64_t i = 0; i < iters; i++) {
            // Compute the block indices
            uint64_t this_block_start = i*block_size;
            uint64_t this_block_end = min(image_length, this_block_start + block_size);
            uint64_t this_block_size = this_block_end-this_block_start;
            voxel_type *buffer = voxels + this_block_start;

            // Copy the block to the GPU
            #pragma acc data copyin(buffer[:this_block_size])
            {
                // Compute the block
                #pragma acc parallel loop
                for (uint64_t j = 0; j < this_block_size; j++) {
                    uint64_t flat_idx = i*block_size + j;
                    voxel_type voxel = buffer[j];
                    voxel = (voxel >= vmin && voxel <= vmax) ? voxel: 0; // Mask away voxels that are not in specified range

                    if VALID_VOXEL(voxel) { // Voxel not masked, and within vmin,vmax range
                        uint64_t x = flat_idx % Nx;
                        uint64_t y = (flat_idx / Nx) % Ny;
                        uint64_t z = (flat_idx / (Nx*Ny)) + z_start;
                        uint64_t r = floor(sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy)));

                        int64_t voxel_index = floor(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                        #pragma acc atomic
                        ++x_bins[x*voxel_bins + voxel_index];
                        #pragma acc atomic
                        ++y_bins[y*voxel_bins + voxel_index];
                        #pragma acc atomic
                        ++z_bins[z*voxel_bins + voxel_index];
                        #pragma acc atomic
                        ++r_bins[r*voxel_bins + voxel_index];
                    }
                }
            }
        }
    }

    auto end = chrono::steady_clock::now();

    if (verbose) {
        chrono::duration<double> diff = end - start;
        printf("Compute took %.04f seconds\n", diff.count());
        auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
        tm local_tm = *localtime(&now);
        printf("Exited function at %02d:%02d:%02d\n", local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
        fflush(stdout);
    }
#else
    throw runtime_error("Library wasn't compiled with OpenACC.");
#endif
}

// On entry, np_*_bins are assumed to be pre allocated and zeroed.
// This function is kept for verification
void axis_histogram_seq_cpu(const py::array_t<voxel_type> np_voxels,
                    const tuple<uint64_t,uint64_t,uint64_t> offset,
                    const uint64_t block_size,
                    py::array_t<uint64_t> &np_x_bins,
                    py::array_t<uint64_t> &np_y_bins,
                    py::array_t<uint64_t> &np_z_bins,
                    py::array_t<uint64_t> &np_r_bins,
                    const tuple<uint64_t, uint64_t> center,
                    const tuple<double, double> vrange,
                    const bool verbose) {

    if (verbose) {
        auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
        tm local_tm = *localtime(&now);
        printf("Entered function at %02d:%02d:%02d\n", local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
    }

    py::buffer_info
        voxels_info = np_voxels.request(),
        x_info = np_x_bins.request(),
        y_info = np_y_bins.request(),
        z_info = np_z_bins.request(),
        r_info = np_r_bins.request();

    const uint64_t
        voxel_bins = x_info.shape[1],
        Nx = x_info.shape[0],
        Ny = y_info.shape[0],
        Nz = z_info.shape[0],
        Nr = r_info.shape[0];

    uint64_t
        *x_bins = static_cast<uint64_t*>(x_info.ptr),
        *y_bins = static_cast<uint64_t*>(y_info.ptr),
        *z_bins = static_cast<uint64_t*>(z_info.ptr),
        *r_bins = static_cast<uint64_t*>(r_info.ptr);

    auto [cy, cx] = center;
    auto [vmin, vmax] = vrange;
    auto [z_start, y_start, x_start] = offset;
    uint64_t
        z_end   = min(z_start+block_size, Nz),
        y_end   = Ny,
        x_end   = Nx;

    const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);

    if (verbose) {
        printf("\nStarting winning %p: (vmin,vmax) = (%g,%g), (Nx,Ny,Nz,Nr) = (%ld,%ld,%ld,%ld)\n",voxels,vmin, vmax, Nx,Ny,Nz,Nr);
        printf("Starting calculation\n");
        fflush(stdout);
    }

    auto start = chrono::steady_clock::now();

    uint64_t flat_idx = 0;
    for (uint64_t z = z_start; z < z_end; z++) {
        for (uint64_t y = y_start; y < y_end; y++) {
            for (uint64_t x = x_start; x < x_end; x++) {
                uint64_t r = floor(sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy)));

                auto voxel = voxels[flat_idx];
                voxel = (voxel >= vmin && voxel <= vmax) ? voxel: 0; // Mask away voxels that are not in specified range
                int64_t voxel_index = round(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                if (voxel_index >= (int64_t)voxel_bins) {
                    fprintf(stderr,"Out-of-bounds error for index %ld: %ld > %ld:\n", flat_idx, voxel_index, voxel_bins);
                } else if VALID_VOXEL(voxel) { // Voxel not masked, and within vmin,vmax range
                    x_bins[x*voxel_bins + voxel_index]++;
                    y_bins[y*voxel_bins + voxel_index]++;
                    z_bins[z*voxel_bins + voxel_index]++;
                    r_bins[r*voxel_bins + voxel_index]++;
                }
                flat_idx++;
            }
        }
    }

    auto end = chrono::steady_clock::now();

    if (verbose) {
        chrono::duration<double> diff = end - start;
        printf("Compute took %.04f seconds\n", diff.count());
        auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
        tm local_tm = *localtime(&now);
        printf("Exited function at %02d:%02d:%02d\n", local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
        fflush(stdout);
    }
}

// TODO: Allow field to be lower resolution than voxel data
void field_histogram_seq_cpu(const py::array_t<voxel_type> np_voxels,
                             const py::array_t<field_type> np_field,
                             const tuple<uint64_t,uint64_t,uint64_t> offset,
                             const tuple<uint64_t,uint64_t,uint64_t> voxels_shape,
                             const tuple<uint64_t,uint64_t,uint64_t> field_shape,
                             const uint64_t block_size,
                             py::array_t<uint64_t> &np_bins,
                             const tuple<double, double> vrange,
                             const tuple<double, double> frange) {
    py::buffer_info
        voxels_info = np_voxels.request(),
        field_info = np_field.request(),
        bins_info = np_bins.request();

    const uint64_t
        field_bins   = bins_info.shape[0],
        voxel_bins   = bins_info.shape[1];

    auto [nZ, nY, nX] = voxels_shape;
    auto [nz, ny, nx] = field_shape;

    double dz = nz/((double)nZ), dy = ny/((double)nY), dx = nx/((double)nX);

    const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);
    const field_type *field  = static_cast<field_type*>(field_info.ptr);
    uint64_t *bins = static_cast<uint64_t*>(bins_info.ptr);

    auto [f_min, f_max] = frange;
    auto [v_min, v_max] = vrange;
    auto [z_start, y_start, x_start] = offset;
    uint64_t
        z_end = min(z_start+block_size, nZ),
        y_end = nY,
        x_end = nX;

    uint64_t flat_index = 0;
    for (uint64_t Z = 0; Z < z_end-z_start; Z++) {
        for (uint64_t Y = y_start; Y < y_end; Y++) {
            for (uint64_t X = x_start; X < x_end; X++) {
                auto voxel = voxels[flat_index];
                voxel = (voxel >= v_min && voxel <= v_max) ? voxel : 0; // Mask away voxels that are not in specified range
                int64_t voxel_index = floor(static_cast<double>(voxel_bins-1) * ((voxel - v_min)/(v_max - v_min)) );

                // What are the X,Y,Z indices corresponding to voxel basearray index I?
                //uint64_t X = flat_index % nX, Y = (flat_index / nX) % nY, Z = (flat_index / (nX*nY)) + z_start;

                // And what are the corresponding x,y,z coordinates into the field array, and field basearray index i?
                // TODO: Sample 2x2x2 volume?
                uint64_t x = floor(X*dx), y = floor(Y*dy), z = floor(Z*dz);
                uint64_t i = z*ny*nx + y*nx + x;

                // TODO the last row of the histogram does not work, when the mask is "bright". Should be discarded.
                if(VALID_VOXEL(voxel) && (field[i]>0)){ // Mask zeros in both voxels and field (TODO: should field be masked, or 0 allowed?)
                    int64_t field_index = floor(static_cast<double>(field_bins-1) * ((field[i] - f_min)/(f_max - f_min)) );

                    bins[field_index*voxel_bins + voxel_index]++;
                }
                flat_index++;
            }
        }
    }
}

void field_histogram_par_cpu(const py::array_t<voxel_type> np_voxels,
                             const py::array_t<field_type> np_field,
                             const tuple<uint64_t,uint64_t,uint64_t> offset,
                             const tuple<uint64_t,uint64_t,uint64_t> voxels_shape,
                             const tuple<uint64_t,uint64_t,uint64_t> field_shape,
                             const uint64_t block_size,
                             py::array_t<uint64_t> &np_bins,
                             const tuple<double, double> vrange,
                             const tuple<double, double> frange) {
    py::buffer_info
        voxels_info = np_voxels.request(),
        field_info = np_field.request(),
        bins_info = np_bins.request();

    const uint64_t
        bins_length  = bins_info.size,
        field_bins   = bins_info.shape[0],
        voxel_bins   = bins_info.shape[1];

    auto [nZ, nY, nX] = voxels_shape;
    auto [nz, ny, nx] = field_shape;

    double dz = nz/((double)nZ), dy = ny/((double)nY), dx = nx/((double)nX);

    const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);
    const field_type *field  = static_cast<field_type*>(field_info.ptr);
    uint64_t *bins = static_cast<uint64_t*>(bins_info.ptr);

    auto [f_min, f_max] = frange;
    auto [v_min, v_max] = vrange;
    auto [z_start, y_start, x_start] = offset;
    uint64_t
        z_end = min(z_start+block_size, nZ),
        y_end = nY,
        x_end = nX;

    #pragma omp parallel
    {
        uint64_t *tmp_bins = (uint64_t*) malloc(sizeof(uint64_t) * bins_length);
        #pragma omp for nowait
        for (uint64_t Z = 0; Z < z_end-z_start; Z++) {
            for (uint64_t Y = y_start; Y < y_end; Y++) {
                for (uint64_t X = x_start; X < x_end; X++) {
                    uint64_t flat_index = (Z*nY*nX) + (Y*nX) + X;
                    auto voxel = voxels[flat_index];
                    voxel = (voxel >= v_min && voxel <= v_max) ? voxel: 0; // Mask away voxels that are not in specified range
                    int64_t voxel_index = floor(static_cast<double>(voxel_bins-1) * ((voxel - v_min)/(v_max - v_min)) );

                    // And what are the corresponding x,y,z coordinates into the field array, and field basearray index i?
                    // TODO: Sample 2x2x2 volume?
                    uint64_t x = floor(X*dx), y = floor(Y*dy), z = floor(Z*dz);
                    uint64_t i = z*ny*nx + y*nx + x;

                    // TODO the last row of the histogram does not work, when the mask is "bright". Should be discarded.
                    if(VALID_VOXEL(voxel) && (field[i] > 0)) { // Mask zeros in both voxels and field (TODO: should field be masked, or 0 allowed?)
                        int64_t field_index = floor(static_cast<double>(field_bins-1) * ((field[i] - f_min)/(f_max - f_min)) );

                        tmp_bins[field_index*voxel_bins + voxel_index]++;
                    }
                }
            }
        }
        #pragma omp critical
        {
            for (uint64_t i = 0; i < bins_length; i++)
                bins[i] += tmp_bins[i];
        }
        free(tmp_bins);
    }
}

INLINE float resample2x2x2(const field_type *voxels,
			   const tuple<uint64_t,uint64_t,uint64_t> &shape,
			   const array<float,3>    &X)
{
      auto  [Nz,Ny,Nx] = shape;	// Eller omvendt?
      assert(X[0]>=0.5      && X[1]>=0.5      && X[2]>= 0.5);
      assert(X[0]<=(Nx-0.5) && X[1]<=(Ny-0.5) && X[2]<= (Nz-0.5));

      float   Xfrac[2][3];	// {Xminus[3], Xplus[3]}
      int64_t  Xint[2][3];	// {Iminus[3], Iplus[3]}
      float   value = 0;

      for(int i=0;i<3;i++){
	double Iminus, Iplus;
	Xfrac[0][i] = 1-modf(X[i]-0.5, &Iminus); // 1-{X[i]-1/2}, floor(X[i]-1/2)
	Xfrac[1][i] =   modf(X[i]+0.5, &Iplus);  // {X[i]+1/2}, floor(X[i]+1/2)

	Xint[0][i] = Iminus;
	Xint[1][i] = Iplus;
      }

      // Resample voxel in 2x2x2 neighbourhood
      //000 ---
      //001 --+
      //010 -+-
      //011 -++
      //100 +--
      //101 +-+
      //110 ++-
      //111 +++

      for(int ijk=0; ijk<=7; ijk++) {
	float  weight = 1;
	int64_t IJK[3] = {0,0,0};

	for(int axis=0;axis<3;axis++){ // x-1/2 or x+1/2
	  int pm = (ijk>>axis) & 1;
	  IJK[axis] = Xint[pm][axis];
	  weight   *= Xfrac[pm][axis];
	}

	auto [I,J,K] = IJK;
	if(I<0 || J<0 || K<0){
	  printf("(I,J,K) = (%ld,%ld,%ld)\n",I,J,K);

	  abort();
	}
	if(I>=int(Nx) || J>=int(Ny) || K>=int(Nz)){
	  printf("(I,J,K) = (%ld,%ld,%ld), (Nx,Ny,Nz) = (%ld,%ld,%ld)\n",I,J,K,Nx,Ny,Nz);
	  abort();
	}
	uint64_t voxel_index = I+J*Nx+K*Nx*Ny;
	field_type voxel = voxels[voxel_index];
	value += voxel*weight;
      }
      return value;
    }

void field_histogram_resample_par_cpu(const py::array_t<voxel_type> np_voxels,
					  const py::array_t<field_type> np_field,
					  const tuple<uint64_t,uint64_t,uint64_t> offset,
				          const tuple<uint64_t,uint64_t,uint64_t> voxels_shape,
				          const tuple<uint64_t,uint64_t,uint64_t> field_shape,
					  const uint64_t block_size,
					  py::array_t<uint64_t> &np_bins,
					  const tuple<double, double> vrange,
					  const tuple<double, double> frange) {
    py::buffer_info
        voxels_info = np_voxels.request(),
        field_info = np_field.request(),
        bins_info = np_bins.request();

    const uint64_t
        bins_length  = bins_info.size,
        field_bins   = bins_info.shape[0],
        voxel_bins   = bins_info.shape[1];

    auto [nZ, nY, nX] = voxels_shape;
    auto [nz, ny, nx] = field_shape;

    float dz = nz/((float)nZ), dy = ny/((float)nY), dx = nx/((float)nX);

    const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);
    const field_type *field  = static_cast<field_type*>(field_info.ptr);
    uint64_t *bins = static_cast<uint64_t*>(bins_info.ptr);

    auto [f_min, f_max] = frange;
    auto [v_min, v_max] = vrange;
    auto [z_start, y_start, x_start] = offset;
    uint64_t
        z_end = min(z_start+block_size, nZ),
        y_end = nY,
        x_end = nX;

    #pragma omp parallel
    {
        uint64_t *tmp_bins = (uint64_t*) malloc(sizeof(uint64_t) * bins_length);
        #pragma omp for nowait collapse(3)
        for (uint64_t Z = 0; Z < z_end-z_start; Z++) {
            for (uint64_t Y = y_start; Y < y_end; Y++) {
                for (uint64_t X = x_start; X < x_end; X++) {
                    uint64_t flat_index = (Z*nY*nX) + (Y*nX) + X;
                    auto voxel = voxels[flat_index];
                    voxel = (voxel >= v_min && voxel <= v_max) ? voxel: 0; // Mask away voxels that are not in specified range
                    int64_t voxel_index = floor(static_cast<float>(voxel_bins-1) * ((voxel - v_min)/(v_max - v_min)) );

                    // And what are the corresponding x,y,z coordinates into the field array?
                    array<float,3> xyz = {X*dx, Y*dy, Z*dz};
		    auto [x,y,z] = xyz;
		    uint16_t  field_value = 0;
		    if(x>=0.5 && y>=0.5 && z>=0.5 && (x+0.5)<nx && (y+0.5)<ny && (z+0.5)<nz)
		      field_value = round(resample2x2x2(field,field_shape,xyz));
		    else {
		      uint64_t i = floor(z)*ny*nx + floor(y)*nx + floor(x);
		      field_value = field[i];
		    }

                    // TODO the last row of the histogram does not work, when the mask is "bright". Should be discarded.
                    if(VALID_VOXEL(voxel) && (field_value > 0)) { // Mask zeros in both voxels and field (TODO: should field be masked, or 0 allowed?)
                        int64_t field_index = floor(static_cast<double>(field_bins-1) * ((field_value - f_min)/(f_max - f_min)) );

                        tmp_bins[field_index*voxel_bins + voxel_index]++;
                    }
                }
            }
        }
        #pragma omp critical
        {
            for (uint64_t i = 0; i < bins_length; i++)
                bins[i] += tmp_bins[i];
        }
        free(tmp_bins);
    }


}

PYBIND11_MODULE(histograms, m) {
    m.doc() = "2D histogramming plugin"; // optional module docstring
    m.def("load_slice", &load_slice);
    m.def("append_slice", &append_slice);
    m.def("write_slice", &write_slice);
    m.def("axis_histogram_seq_cpu",  &axis_histogram_seq_cpu);
    m.def("axis_histogram_par_cpu",  &axis_histogram_par_cpu);
    m.def("axis_histogram_par_gpu",  &axis_histogram_par_gpu);
    m.def("field_histogram_seq_cpu", &field_histogram_seq_cpu);
    m.def("field_histogram_par_cpu", &field_histogram_par_cpu);
    m.def("field_histogram_resample_par_cpu", &field_histogram_resample_par_cpu);
    m.def("masked_minmax", &masked_minmax);
    m.def("float_minmax", &float_minmax);
    m.def("gauss_filter_par_cpu", &gauss_filter_par_cpu);
}
