#include "histograms.hh"

namespace gpu {

    // On entry, np_*_bins are assumed to be pre allocated and zeroed.
    void axis_histogram(const voxel_type __restrict__* voxels,
                        const shape_t &voxels_shape,
                        const shape_t &offset,
                        const shape_t &block_size,
                        uint64_t __restrict__* x_bins,
                        uint64_t __restrict__* y_bins,
                        uint64_t __restrict__* z_bins,
                        uint64_t __restrict__* r_bins,
                        const uint64_t voxel_bins,
                        const uint64_t Nr,
                        const std::tuple<uint64_t, uint64_t> &center,
                        const std::tuple<double, double> &vrange,
                        const bool verbose) {
    #ifdef _OPENACC
        if (verbose) {
            auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            tm local_tm = *localtime(&now);
            printf("Entered function at %02d:%02d:%02d\n", local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
        }

        auto [Nz, Ny, Nx] = voxels_shape;
        auto [cy, cx] = center;
        auto [vmin, vmax] = vrange;
        auto [z_start, y_start, x_start] = offset;

        uint64_t memory_needed = ((Nx*voxel_bins)+(Ny*voxel_bins)+(Nz*voxel_bins)+(Nr*voxel_bins))*sizeof(uint64_t);

        uint64_t
            z_end   = (uint64_t) std::min(z_start+block_size.z, Nz),
            //y_end   = Ny,
            //x_end   = Nx,
            image_length = block_size.z*block_size.y*block_size.x;

        uint64_t gpu_block_size = 1 * GB_VOXEL;

        uint64_t n_iterations = (image_length + (gpu_block_size-1)) / gpu_block_size;

        uint64_t initial_block = std::min(image_length, gpu_block_size);

        if (verbose) {
            printf("\nStarting %p: (vmin,vmax) = (%g,%g), (Nx,Ny,Nz,Nr) = (%ld,%ld,%ld,%ld)\n",voxels,vmin, vmax, Nx,Ny,Nz,Nr);
            printf("Offset is (%ld,%ld,%ld)\n", z_start, y_start, x_start);
            printf("Allocating result memory (%ld bytes (%.02f Mbytes))\n", memory_needed, memory_needed/1024./1024.);
            printf("Starting calculation\n");
            printf("Size of voxels is %ld bytes (%.02f Mbytes)\n", image_length * sizeof(voxel_type), (image_length * sizeof(voxel_type))/1024./1024.);
            printf("Blocksize is %ld bytes (%.02f Mbytes)\n", gpu_block_size * sizeof(voxel_type), (gpu_block_size * sizeof(voxel_type))/1024./1024.);
            printf("Block shape is (%ld,%ld,%ld)\n", block_size.z, block_size.y, block_size.x);
            printf("Doing %d blocks\n", n_iterations);
            fflush(stdout);
        }

        auto start = std::chrono::steady_clock::now();

        // Copy the buffers to the GPU on entry and back to host on exit
        #pragma acc data copy(x_bins[:Nx*voxel_bins], y_bins[:Ny*voxel_bins], z_bins[:Nz*voxel_bins], r_bins[:Nr*voxel_bins])
        {
            // For each block
            for (uint64_t i = 0; i < n_iterations; i++) {
                // Compute the block indices
                uint64_t this_block_start = i*gpu_block_size;
                uint64_t this_block_end = std::min(image_length, this_block_start + gpu_block_size);
                uint64_t this_block_size = this_block_end-this_block_start;
                const voxel_type *buffer = voxels + this_block_start;

                // Copy the block to the GPU
                #pragma acc data copyin(buffer[:this_block_size])
                {
                    // Compute the block
                    #pragma acc parallel loop present(buffer[:this_block_size], x_bins, y_bins, z_bins, r_bins)
                    for (uint64_t j = 0; j < this_block_size; j++) {
                        uint64_t flat_idx = i * gpu_block_size + j;
                        voxel_type voxel = buffer[j];
                        voxel = (voxel >= vmin && voxel <= vmax) ? voxel: 0; // Mask away voxels that are not in specified range

                        if (voxel != 0) { // Voxel not masked, and within vmin,vmax range
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

        auto end = std::chrono::steady_clock::now();

        if (verbose) {
            std::chrono::duration<double> diff = end - start;
            printf("Compute took %.04f seconds\n", diff.count());
            auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            tm local_tm = *localtime(&now);
            printf("Exited function at %02d:%02d:%02d\n", local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
            fflush(stdout);
        }
        #else
            throw std::runtime_error("Library wasn't compiled with OpenACC.");
        #endif
    }

    void field_histogram(const voxel_type *__restrict__ voxels,
                         const field_type *__restrict__ field,
                         const shape_t &voxels_shape,
                         const shape_t &field_shape,
                         const shape_t &offset,
                         const shape_t &block_size,
                         uint64_t *__restrict__ bins,
                         const uint64_t voxel_bins,
                         const uint64_t field_bins,
                         const std::tuple<double, double> &vrange,
                         const std::tuple<double, double> &frange,
                         const bool verbose) {
        if (verbose) {
            auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            tm local_tm = *localtime(&now);
            printf("Entered function at %02d:%02d:%02d\n", local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
        }

        auto [nZ, nY, nX] = voxels_shape;
        auto [nz, ny, nx] = field_shape;

        if (verbose) {
            printf("voxels shape: %ld %ld %ld\n", nZ, nY, nX);
            printf("field shape: %ld %ld %ld\n", nz, ny, nx);
        }

        double
            dz = (double)nz/((double)nZ),
            dy = (double)ny/((double)nY),
            dx = (double)nx/((double)nX);

        auto [f_min, f_max] = frange;
        auto [v_min, v_max] = vrange;
        auto [z_start, y_start, x_start] = offset;
        uint64_t
            image_length = block_size.z*nY*nX,
            bins_length = field_bins * voxel_bins,
            z_end = (uint64_t) std::min(z_start+block_size.z, nZ),
            //y_end = nY,
            //x_end = nX,
            flat_factor = (uint64_t) (1/dz) * (uint64_t) (1/dy) * (uint64_t) (1/dx);

        uint64_t gpu_block_size = 1 * GB_VOXEL;

        uint64_t n_iterations = (image_length + gpu_block_size-1) / gpu_block_size;

        if (verbose) {
            printf("voxels shape: (%ld,%ld,%ld)\n", nZ, nY, nX);
            printf("field shape: (%ld,%ld,%ld)\n", nz, ny, nx);
            printf("offset: (%ld,%ld,%ld)\n", z_start, y_start, x_start);
            printf("block_size: (%ld,%ld,%ld)\n", block_size.z, block_size.y, block_size.x);
            printf("vrange: (%g,%g)\n", v_min, v_max);
            printf("frange: (%g,%g)\n", f_min, f_max);
            printf("deltas: (%g,%g,%g)\n", dz, dy, dx);
            printf("n_iterations: %ld\n", n_iterations);
            printf("image_length: %ld\n", image_length);
            printf("gpu_block_size: %ld\n", gpu_block_size);
            printf("bins_length: %ld\n", bins_length);
            printf("flat_factor: %ld\n", flat_factor);
        }

        auto start = std::chrono::steady_clock::now();

        #pragma acc data copy(bins[:bins_length])
        {
            // For each block
            for (uint64_t iter = 0; iter < n_iterations; iter++) {
                // Compute the block indices
                uint64_t this_block_start = iter*gpu_block_size;
                uint64_t this_block_end = std::min(image_length, this_block_start + gpu_block_size);
                uint64_t this_block_size = this_block_end-this_block_start;
                const voxel_type *voxels_buffer = voxels + this_block_start;
                const field_type *field_buffer = field + (this_block_start / flat_factor); // TODO handle field of different size

                // Copy the block to the GPU
                #pragma acc data copyin(voxels_buffer[:this_block_size], field_buffer[:(this_block_size/flat_factor)])
                {
                    #pragma acc parallel loop
                    for (uint64_t flat_index = 0; flat_index < this_block_size; flat_index++) {
                        uint64_t
                            Z = (flat_index / (nX*nY)),
                            Y = (flat_index / nX) % nY,
                            X = flat_index % nX;
                        voxel_type voxel = voxels_buffer[flat_index];
                        voxel = (voxel >= v_min && voxel <= v_max) ? voxel : 0; // Mask away voxels that are not in specified range
                        int64_t voxel_index = (int64_t) floor(static_cast<double>(voxel_bins-1) * ((voxel - v_min)/(v_max - v_min)) );

                        // What are the X,Y,Z indices corresponding to voxel basearray index I?
                        //uint64_t X = flat_index % nX, Y = (flat_index / nX) % nY, Z = (flat_index / (nX*nY)) + z_start;

                        // And what are the corresponding x,y,z coordinates into the field array, and field basearray index i?
                        // TODO: Sample 2x2x2 volume?
                        uint64_t
                            z = (uint64_t) floor((double)Z*dz),
                            y = (uint64_t) floor((double)Y*dy),
                            x = (uint64_t) floor((double)X*dx);
                        uint64_t i = z*ny*nx + y*nx + x;
                        field_type field_value = field_buffer[i];

                        // TODO the last row of the histogram does not work, when the mask is "bright". Should be discarded.
                        if ((voxel != 0) && (field_value > 0)) { // Mask zeros in both voxels and field (TODO: should field be masked, or 0 allowed?)
                            int64_t field_index = (int64_t) floor(static_cast<double>(field_bins-1) * ((field_value - f_min)/(f_max - f_min)) );

                            #pragma acc atomic
                            bins[field_index*voxel_bins + voxel_index]++;
                        }
                    }
                }
            }
        }

        auto end = std::chrono::steady_clock::now();

        if (verbose) {
            std::chrono::duration<double> diff = end - start;
            printf("Compute took %.04f seconds\n", diff.count());
            auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            tm local_tm = *localtime(&now);
            printf("Exited function at %02d:%02d:%02d\n", local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
            fflush(stdout);
        }
    }

    void field_histogram_resample(const voxel_type *__restrict__ voxels,
                                  const field_type *__restrict__ field,
                                  const shape_t &voxels_shape,
                                  const shape_t &field_shape,
                                  const shape_t &offset,
                                  const shape_t &block_size,
                                  uint64_t *__restrict__ &bins,
                                  const uint64_t voxel_bins,
                                  const uint64_t field_bins,
                                  const std::tuple<double, double> &vrange,
                                  const std::tuple<double, double> &frange,
                                  const bool verbose) {
        throw std::runtime_error("Not implemented");
    }

}