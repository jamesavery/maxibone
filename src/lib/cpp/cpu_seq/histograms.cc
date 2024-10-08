/**
 * @file histograms.cc
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief CPU sequential implementations of the 2D histogram functions.
 * @version 0.1
 * @date 2024-09-17
 *
 * @copyright Copyright (c) 2024
 */
#include "histograms.hh"

// These are the original implementations, which are kept for verification purposes.

namespace cpu_seq {

    void axis_histogram(const voxel_type __restrict__* voxels,
                        const shape_t &global_shape,
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
                        const int verbose) {

        if (verbose >= 2) {
            auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            tm local_tm = *localtime(&now);
            printf("Entered function at %02d:%02d:%02d\n", local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
        }

        auto [Nz, Ny, Nx] = global_shape;
        auto [cy, cx] = center;
        auto [vmin, vmax] = vrange;
        auto [z_start, y_start, x_start] = offset;

        uint64_t
            z_end   = (uint64_t) std::min(z_start+block_size.z, Nz),
            //z_size  = z_end - z_start,
            y_end   = Ny,
            x_end   = Nx;

        if (verbose >= 2) {
            printf("\nStarting %p: (vmin,vmax) = (%g,%g), (Nx,Ny,Nz,Nr) = (%ld,%ld,%ld,%ld)\n", voxels, vmin, vmax, Nx, Ny, Nz, Nr);
            printf("Starting calculation\n");
            fflush(stdout);
        }

        auto start = std::chrono::steady_clock::now();

        uint64_t flat_idx = 0*Ny*Nx + y_start*Nx + x_start;
        for (uint64_t z = z_start; z < z_end; z++) {
            for (uint64_t y = y_start; y < y_end; y++) {
                for (uint64_t x = x_start; x < x_end; x++) {
                    uint64_t r = (uint64_t) std::floor(std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy)));

                    auto voxel = voxels[flat_idx];
                    voxel = (voxel >= vmin && voxel <= vmax) ? voxel : 0; // Mask away voxels that are not in specified range
                    int64_t voxel_index = (int64_t) std::floor(static_cast<double>(voxel_bins-1) * ((voxel - vmin) / (vmax - vmin)) );

                    if (voxel_index >= (int64_t)voxel_bins) {
                        fprintf(stderr,"Out-of-bounds error for index %ld: %ld > %ld:\n", flat_idx, voxel_index, voxel_bins);
                    } else if (voxel != 0) { // Voxel not masked, and within vmin,vmax range
                        x_bins[x*voxel_bins + voxel_index]++;
                        y_bins[y*voxel_bins + voxel_index]++;
                        z_bins[z*voxel_bins + voxel_index]++;
                        r_bins[r*voxel_bins + voxel_index]++;
                    }
                    flat_idx++;
                }
            }
        }

        auto end = std::chrono::steady_clock::now();

        if (verbose >= 2) {
            std::chrono::duration<double> diff = end - start;
            printf("Compute took %.04f seconds\n", diff.count());
            auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            tm local_tm = *localtime(&now);
            printf("Exited function at %02d:%02d:%02d\n", local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
            fflush(stdout);
        }
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
                         const int verbose) {

        auto [nZ, nY, nX] = voxels_shape;
        auto [nz, ny, nx] = field_shape;

        double
            dz = (double)nz/((double)nZ),
            dy = (double)ny/((double)nY),
            dx = (double)nx/((double)nX);

        auto [f_min, f_max] = frange;
        auto [v_min, v_max] = vrange;
        auto [z_start, y_start, x_start] = offset;
        uint64_t
            //z_end = (uint64_t) std::min(z_start+block_size.z, nZ),
            y_end = nY,
            x_end = nX;

        if (verbose >= 2) {
            printf("\nStarting calculation\n");
            printf("nZ, nY, nX = %ld, %ld, %ld\n", nZ, nY, nX);
            printf("nz, ny, nx = %ld, %ld, %ld\n", nz, ny, nx);
            printf("dz, dy, dx = %g, %g, %g\n", dz, dy, dx);
            printf("vrange = (%g, %g), frange = (%g, %g)\n", v_min, v_max, f_min, f_max);
            fflush(stdout);
        }

        uint64_t flat_index = 0;
        for (uint64_t Z = 0; Z < (uint64_t) block_size.z; Z++) {
            for (uint64_t Y = y_start; Y < y_end; Y++) {
                for (uint64_t X = x_start; X < x_end; X++) {
                    auto voxel = voxels[flat_index];
                    voxel = (voxel >= v_min && voxel <= v_max) ? voxel : 0; // Mask away voxels that are not in specified range
                    int64_t voxel_index = (int64_t) std::floor(static_cast<double>(voxel_bins-1) * ((voxel - v_min) / (v_max - v_min)) );

                    // What are the X,Y,Z indices corresponding to voxel basearray index I?
                    //uint64_t X = flat_index % nX, Y = (flat_index / nX) % nY, Z = (flat_index / (nX*nY)) + z_start;

                    // And what are the corresponding x,y,z coordinates into the field array, and field basearray index i?
                    // TODO: Sample 2x2x2 volume?
                    uint64_t
                        z = (uint64_t) std::floor((double)Z*dz),
                        y = (uint64_t) std::floor((double)Y*dy),
                        x = (uint64_t) std::floor((double)X*dx);
                    uint64_t i = z*ny*nx + y*nx + x;

                    // TODO the last row of the histogram does not work, when the mask is "bright". Should be discarded.
                    if((voxel != 0) && (field[i] > 0)){ // Mask zeros in both voxels and field (TODO: should field be masked, or 0 allowed?)
                        int64_t field_index = (int64_t) std::floor(static_cast<double>(field_bins-1) * ((field[i] - f_min) / (f_max - f_min)) );

                        bins[field_index*voxel_bins + voxel_index]++;
                    }
                    flat_index++;
                }
            }
        }
    }

    #pragma GCC diagnostic ignored "-Wunused-parameter"
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
                                  const int verbose) {
        throw std::runtime_error("Not implemented");
    }

}