#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <inttypes.h>
#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <iostream>
using namespace std;
namespace py = pybind11;

typedef uint16_t voxel_type;
typedef float    field_type;

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

// On entry, np_*_bins are assumed to be pre allocated and zeroed.
void axis_histogram_par_cpu(const py::array_t<voxel_type> np_voxels,
                            py::array_t<uint64_t> &np_x_bins,
                            py::array_t<uint64_t> &np_y_bins,
                            py::array_t<uint64_t> &np_z_bins,
                            py::array_t<uint64_t> &np_r_bins,
                            const double vmin, const double vmax,
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
        for (uint64_t x = 0; x < Nx; x++) {
            // Init
            #pragma omp simd
            for (uint64_t i = 0; i < voxel_bins; i++)
                tmp[i] = 0;

            // Read & Compute
            #pragma omp simd collapse(2)
            for (uint64_t z = 0; z < Nz; z++) {
                for (uint64_t y = 0; y < Ny; y++) {
                    uint64_t flat_idx = z*Ny*Nx + y*Nx + x;
                    auto voxel = voxels[flat_idx];
                    int64_t voxel_index = floor(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                    if (voxel_index >= (int64_t) voxel_bins) {
                        fprintf(stderr,"Out-of-bounds error for index %ld: %ld > %ld:\n", flat_idx, voxel_index, voxel_bins);
                    } else if(voxel != 0) {
                        tmp[voxel_index]++;
                    }
                }
            }

            // Store
            #pragma omp simd
            for (uint64_t i = 0; i < voxel_bins; i++)
                x_bins[x*voxel_bins + i] = tmp[i];
        }

        // y_bins
        #pragma omp for nowait
        for (uint64_t y = 0; y < Ny; y++) {
            // Init
            #pragma omp simd
            for (uint64_t i = 0; i < voxel_bins; i++)
                tmp[i] = 0;

            // Read & Compute
            #pragma omp simd collapse(2)
            for (uint64_t z = 0; z < Nz; z++) {
                for (uint64_t x = 0; x < Nx; x++) {
                    uint64_t flat_idx = z*Ny*Nx + y*Nx + x;
                    auto voxel = voxels[flat_idx];
                    int64_t voxel_index = floor(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                    if (voxel_index >= (int64_t) voxel_bins) {
                        fprintf(stderr,"Out-of-bounds error for index %ld: %ld > %ld:\n", flat_idx, voxel_index, voxel_bins);
                    } else if(voxel != 0) {
                        tmp[voxel_index]++;
                    }
                }
            }

            // Store
            #pragma omp simd
            for (uint64_t i = 0; i < voxel_bins; i++)
                y_bins[y*voxel_bins + i] = tmp[i];
        }

        // r_bins
        #pragma omp for nowait
        for (uint64_t z = 0; z < Nz; z++) {
            // Init
            #pragma omp simd
            for (uint64_t i = 0; i < voxel_bins; i++)
                tmp[i] = 0;

            // Read & Compute
            #pragma omp simd collapse(2)
            for (uint64_t y = 0; y < Ny; y++) {
                for (uint64_t x = 0; x < Nx; x++) {
                    uint64_t flat_idx = z*Ny*Nx + y*Nx + x;
                    auto voxel = voxels[flat_idx];
                    int64_t voxel_index = floor(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                    if (voxel_index >= (int64_t) voxel_bins) {
                        fprintf(stderr,"Out-of-bounds error for index %ld: %ld > %ld:\n", flat_idx, voxel_index, voxel_bins);
                    } else if(voxel != 0) {
                        tmp[voxel_index]++;
                    }
                }
            }

            // Store
            #pragma omp simd
            for (uint64_t i = 0; i < voxel_bins; i++)
                z_bins[z*voxel_bins + i] = tmp[i];
        }

        // z_bins
        #pragma omp for nowait collapse(2)
        for (uint64_t y = 0; y < Ny; y++) {
            for (uint64_t x = 0; x < Nx; x++) {
                // Init
                #pragma omp simd
                for (uint64_t i = 0; i < voxel_bins; i++)
                    tmp[i] = 0;

                uint64_t r = floor(sqrt((x-Nx/2.0)*(x-Nx/2.0) + (y-Ny/2.0)*(y-Ny/2.0)));

                // Read and compute
                #pragma omp simd
                for (uint64_t z = 0; z < Nz; z++) {
                    uint64_t flat_idx = z*Ny*Nx + y*Nx + x;
                    auto voxel = voxels[flat_idx];
                    int64_t voxel_index = floor(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                    if (voxel_index >= (int64_t) voxel_bins) {
                        fprintf(stderr,"Out-of-bounds error for index %ld: %ld > %ld:\n", flat_idx, voxel_index, voxel_bins);
                    } else if(voxel != 0) {
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
                            py::array_t<uint64_t> &np_x_bins,
                            py::array_t<uint64_t> &np_y_bins,
                            py::array_t<uint64_t> &np_z_bins,
                            py::array_t<uint64_t> &np_r_bins,
                            const double vmin, const double vmax,
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

    uint64_t block_size =
        512 * 1024 * 1024; // 1 GB
        //64 * 1024 * 1024; // 128 MB

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

                    uint64_t x = flat_idx % Nx;
                    uint64_t y = (flat_idx / Nx) % Ny;
                    uint64_t z = flat_idx / (Nx*Ny);
                    uint64_t r = floor(sqrt((x-Nx/2.0)*(x-Nx/2.0) + (y-Ny/2.0)*(y-Ny/2.0)));

                    int64_t voxel_index = floor(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                    if(voxel != 0) {
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
                    py::array_t<uint64_t> &np_x_bins,
                    py::array_t<uint64_t> &np_y_bins,
                    py::array_t<uint64_t> &np_z_bins,
                    py::array_t<uint64_t> &np_r_bins,
                    const double vmin, const double vmax,
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

    const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);

    if (verbose) {
        printf("\nStarting winning %p: (vmin,vmax) = (%g,%g), (Nx,Ny,Nz,Nr) = (%ld,%ld,%ld,%ld)\n",voxels,vmin, vmax, Nx,Ny,Nz,Nr);
        printf("Starting calculation\n");
        fflush(stdout);
    }

    auto start = chrono::steady_clock::now();

    uint64_t flat_idx = 0;
    for (uint64_t z = 0; z < Nz; z++) {
        for (uint64_t y = 0; y < Ny; y++) {
            for (uint64_t x = 0; x < Nx; x++) {
                uint64_t r = floor(sqrt((x-Nx/2.0)*(x-Nx/2.0) + (y-Ny/2.0)*(y-Ny/2.0)));

                auto voxel = voxels[flat_idx++];
                int64_t voxel_index = floor(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                if (voxel_index >= (int64_t)voxel_bins) {
                    fprintf(stderr,"Out-of-bounds error for index %ld: %ld > %ld:\n", flat_idx, voxel_index, voxel_bins);
                } else if (voxel != 0) {
                    x_bins[x*voxel_bins + voxel_index]++;
                    y_bins[y*voxel_bins + voxel_index]++;
                    z_bins[z*voxel_bins + voxel_index]++;
                    r_bins[r*voxel_bins + voxel_index]++;
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
}

// TODO: Allow field to be lower resolution than voxel data
void field_histogram(const py::array_t<voxel_type> np_voxels,
		     const py::array_t<field_type> np_field,
		     py::array_t<uint64_t> &np_bins,
		     const double vmin, const double vmax
	     )
{
  py::buffer_info
    voxels_info = np_voxels.request(),
    field_info = np_field.request(),
    bins_info = np_bins.request();

  const uint64_t
    image_length = voxels_info.size,
    field_length = field_info.size,
    field_bins   = bins_info.shape[0],
    voxel_bins   = bins_info.shape[1];

  const uint64_t
    nZ = voxels_info.shape[0], nY = voxels_info.shape[1], nX = voxels_info.shape[2],
    nz = field_info.shape[0],  ny = field_info.shape[1],  nx = field_info.shape[2];

  double dz = nz/((double)nZ), dy = ny/((double)nY), dx = nx/((double)nX);

  const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);
  const field_type *field  = static_cast<field_type*>(field_info.ptr);
  uint64_t *bins = static_cast<uint64_t*>(bins_info.ptr);

  uint64_t I=0;

  float fmin=1e6, fmax=0;	// TODO: inf, -inf
  for(uint64_t i=0;i<field_length;i++){
    fmin = field[i]>0? min(field[i],fmin) : fmin; // TODO: Should we really mask field zeros?
    fmax = max(field[i],fmax);
  }

  //#pragma omp parallel for private(I) reduction(+:bins[:field_bins*voxel_bins])
  for(I=0;I<image_length;I++){
    int64_t voxel_index = floor(static_cast<double>(voxel_bins-1) * ((voxels[I] - vmin)/(vmax - vmin)) );

    // What are the X,Y,Z indices corresponding to voxel basearray index I?
    uint64_t X = I % nX, Y = (I / nX) % nY, Z = I / (nX*nY);

    // And what are the corresponding x,y,z coordinates into the field array, and field basearray index i?
    // TODO: Sample 2x2x2 volume?
    uint64_t x = floor(X*dx), y = floor(Y*dy), z = floor(Z*dz);
    uint64_t i = z*ny*nx + y*nx + x;

    if((voxels[I]>=1) && (field[i]>0)){ // Mask zeros in both voxels and field (TODO: should field be masked, or 0 allowed?)
      int64_t field_index = floor(static_cast<double>(field_bins-1) * ((field[i] - fmin)/(fmax - fmin)) );

      bins[field_index*voxel_bins + voxel_index]++;
    }
  }
}

PYBIND11_MODULE(histograms, m) {
    m.doc() = "2D histogramming plugin"; // optional module docstring

    m.def("axis_histogram_seq_cpu",  &axis_histogram_seq_cpu);
    m.def("axis_histogram_par_cpu",  &axis_histogram_par_cpu);
    m.def("axis_histogram_par_gpu",  &axis_histogram_par_gpu);
    m.def("field_histogram", &field_histogram);
    m.def("masked_minmax", &masked_minmax);
}
