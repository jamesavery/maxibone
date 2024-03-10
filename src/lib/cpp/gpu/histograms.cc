#include "histograms.hh"

namespace gpu {

    void axis_histogram_par_gpu(const np_array<voxel_type> np_voxels,
                                const tuple<uint64_t,uint64_t,uint64_t> offset,
                                const uint64_t outside_block_size,
                                np_array<uint64_t> &np_x_bins,
                                np_array<uint64_t> &np_y_bins,
                                np_array<uint64_t> &np_z_bins,
                                np_array<uint64_t> &np_r_bins,
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
        z_end   = min(z_start+outside_block_size, Nz);

        auto [vmin, vmax] = vrange;
        auto [cy, cx] = center;

        uint64_t block_size = 1 * GB_VOXEL;

        uint64_t n_iterations = image_length / block_size;
        if (n_iterations * block_size < image_length)
            n_iterations++;

        uint64_t initial_block = min(image_length, block_size);

        if (verbose) {
            printf("\nStarting %p: (vmin,vmax) = (%g,%g), (Nx,Ny,Nz,Nr) = (%ld,%ld,%ld,%ld)\n",voxels,vmin, vmax, Nx,Ny,Nz,Nr);
            printf("Allocating result memory (%ld bytes (%.02f Mbytes))\n", memory_needed, memory_needed/1024./1024.);
            printf("Starting calculation\n");
            printf("Size of voxels is %ld bytes (%.02f Mbytes)\n", image_length * sizeof(voxel_type), (image_length * sizeof(voxel_type))/1024./1024.);
            printf("Blocksize is %ld bytes (%.02f Mbytes)\n", block_size * sizeof(voxel_type), (block_size * sizeof(voxel_type))/1024./1024.);
            printf("Doing %d blocks\n", n_iterations);
            fflush(stdout);
        }

        auto start = chrono::steady_clock::now();

        // Copy the buffers to the GPU on entry and back to host on exit
        #pragma acc data copy(x_bins[:Nx*voxel_bins], y_bins[:Ny*voxel_bins], z_bins[:Nz*voxel_bins], r_bins[:Nr*voxel_bins])
        {
            // For each block
            for (uint64_t i = 0; i < n_iterations; i++) {
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

}