#include <histograms.hh>

namespace cpu_seq {

    // On entry, np_*_bins are assumed to be pre allocated and zeroed.
    // This function is kept for verification
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

        if (verbose) {
            auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            tm local_tm = *localtime(&now);
            printf("Entered function at %02d:%02d:%02d\n", local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
        }

        auto [Nz, Ny, Nx] = voxels_shape;
        auto [cy, cx] = center;
        auto [vmin, vmax] = vrange;
        auto [z_start, y_start, x_start] = offset;

        uint64_t
            z_end   = (uint64_t) std::min(z_start+block_size.z, Nz),
            y_end   = Ny,
            x_end   = Nx;

        if (verbose) {
            printf("\nStarting %p: (vmin,vmax) = (%g,%g), (Nx,Ny,Nz,Nr) = (%lld%lld,%lld,%lld)\n",voxels,vmin, vmax, Nx,Ny,Nz,Nr);
            printf("Starting calculation\n");
            fflush(stdout);
        }

        auto start = std::chrono::steady_clock::now();

        uint64_t flat_idx = 0;
        for (uint64_t z = z_start; z < z_end; z++) {
            for (uint64_t y = y_start; y < y_end; y++) {
                for (uint64_t x = x_start; x < x_end; x++) {
                    uint64_t r = (uint64_t) floor(sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy)));

                    auto voxel = voxels[flat_idx];
                    voxel = (voxel >= vmin && voxel <= vmax) ? voxel: 0; // Mask away voxels that are not in specified range
                    int64_t voxel_index = (int64_t) round(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                    if (voxel_index >= (int64_t)voxel_bins) {
                        fprintf(stderr,"Out-of-bounds error for index %lld: %lld > %lld:\n", flat_idx, voxel_index, voxel_bins);
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

        if (verbose) {
            std::chrono::duration<double> diff = end - start;
            printf("Compute took %.04f seconds\n", diff.count());
            auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            tm local_tm = *localtime(&now);
            printf("Exited function at %02d:%02d:%02d\n", local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
            fflush(stdout);
        }
    }

    // TODO: Allow field to be lower resolution than voxel data
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
                         const std::tuple<double, double> &frange) {

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
            z_end = (uint64_t) std::min(z_start+block_size.z, nZ),
            y_end = nY,
            x_end = nX;

        uint64_t flat_index = 0;
        for (uint64_t Z = 0; Z < z_end-z_start; Z++) {
            for (uint64_t Y = y_start; Y < y_end; Y++) {
                for (uint64_t X = x_start; X < x_end; X++) {
                    auto voxel = voxels[flat_index];
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

                    // TODO the last row of the histogram does not work, when the mask is "bright". Should be discarded.
                    if((voxel != 0) && (field[i]>0)){ // Mask zeros in both voxels and field (TODO: should field be masked, or 0 allowed?)
                        int64_t field_index = (int64_t) floor(static_cast<double>(field_bins-1) * ((field[i] - f_min)/(f_max - f_min)) );

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
                                  const std::tuple<double, double> &frange) {
        throw std::runtime_error("Not implemented");
    }

}