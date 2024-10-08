/**
 * @file histograms.cc
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief CPU parallel implementations of the 2D histogram functions.
 * @version 0.1
 * @date 2024-09-17
 *
 * @copyright Copyright (c) 2024
 */
#include "histograms.hh"

namespace cpu_par {

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
            z_end = (uint64_t) std::min(z_start+block_size.z, Nz),
            y_end = Ny,
            x_end = Nx;

        if (verbose >= 2) {
            printf("\nStarting %p: (vmin,vmax) = (%g,%g), (Nx,Ny,Nz,Nr) = (%ld,%ld,%ld,%ld)\n",voxels,vmin, vmax, Nx,Ny,Nz,Nr);
            printf("Starting calculation\n");
            fflush(stdout);
        }

        auto start = std::chrono::steady_clock::now();

        // TODO This is faster on Intel?
        #pragma omp parallel for collapse(3)
        for (uint64_t z = z_start; z < z_end; z++) {
            for (uint64_t y = y_start; y < y_end; y++) {
                for (uint64_t x = x_start; x < x_end; x++) {
                    uint64_t flat_idx = (z-z_start)*Ny*Nx + y*Nx + x;
                    uint64_t r = (uint64_t) std::floor(std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy)));

                    auto voxel = voxels[flat_idx];
                    voxel = (voxel >= vmin && voxel <= vmax) ? voxel : 0; // Mask away voxels that are not in specified range
                    int64_t voxel_index = (int64_t) std::floor(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                    if (voxel_index >= (int64_t)voxel_bins) {
                        fprintf(stderr,"Out-of-bounds error for index %ld: %ld > %ld:\n", flat_idx, voxel_index, voxel_bins);
                    } else if (voxel != 0) { // Voxel not masked, and within vmin,vmax range
                        #pragma omp atomic
                        x_bins[x*voxel_bins + voxel_index]++;
                        #pragma omp atomic
                        y_bins[y*voxel_bins + voxel_index]++;
                        #pragma omp atomic
                        z_bins[z*voxel_bins + voxel_index]++;
                        #pragma omp atomic
                        r_bins[r*voxel_bins + voxel_index]++;
                    }
                }
            }
        }

        /*
        // TODO This is faster on AMD?
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
                for (uint64_t z = 0; z < z_size; z++) {
                    for (uint64_t y = y_start; y < y_end; y++) {
                        uint64_t flat_idx = z*Ny*Nx + y*Nx + x;
                        auto voxel = voxels[flat_idx];
                        voxel = (voxel >= vmin && voxel <= vmax) ? voxel: 0; // Mask away voxels that are not in specified range
                        int64_t voxel_index = (int64_t) std::floor(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                        if (voxel_index >= (int64_t) voxel_bins) {
                            fprintf(stderr,"Out-of-bounds error for index %ld: %ld > %ld:\n", flat_idx, voxel_index, voxel_bins);
                        } else if (voxel != 0) { // Voxel not masked, and within vmin,vmax range
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
                for (uint64_t z = 0; z < z_size; z++) {
                    for (uint64_t x = x_start; x < x_end; x++) {
                        uint64_t flat_idx = z*Ny*Nx + y*Nx + x;
                        auto voxel = voxels[flat_idx];
                        voxel = (voxel >= vmin && voxel <= vmax) ? voxel: 0; // Mask away voxels that are not in specified range
                        int64_t voxel_index = (int64_t) std::floor(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                        if (voxel_index >= (int64_t) voxel_bins) {
                            fprintf(stderr,"Out-of-bounds error for index %ld: %ld > %ld:\n", flat_idx, voxel_index, voxel_bins);
                        } else if (voxel != 0) { // Voxel not masked, and within vmin,vmax range
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
            for (uint64_t z = 0; z < z_size; z++) {
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
                        int64_t voxel_index = (int64_t) std::floor(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                        if (voxel_index >= (int64_t) voxel_bins) {
                            fprintf(stderr,"Out-of-bounds error for index %ld: %ld > %ld:\n", flat_idx, voxel_index, voxel_bins);
                        } else if (voxel != 0) { // Voxel not masked, and within vmin,vmax range
                            tmp[voxel_index]++;
                        }
                    }
                }

                // Store
                uint64_t z_idx = z + z_start;
                #pragma omp simd
                for (uint64_t i = 0; i < voxel_bins; i++)
                    z_bins[z_idx*voxel_bins + i] += tmp[i];
            }

            // r_bins
            #pragma omp for nowait collapse(2)
            for (uint64_t y = y_start; y < y_end; y++) {
                for (uint64_t x = x_start; x < x_end; x++) {
                    // Init
                    #pragma omp simd
                    for (uint64_t i = 0; i < voxel_bins; i++)
                        tmp[i] = 0;

                    uint64_t r = (uint64_t) std::floor(std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy)));

                    // Read and compute
                    #pragma omp simd
                    for (uint64_t z = 0; z < z_size; z++) {
                        uint64_t flat_idx = z*Ny*Nx + y*Nx + x;
                        auto voxel = voxels[flat_idx];
                        voxel = (voxel >= vmin && voxel <= vmax) ? voxel: 0; // Mask away voxels that are not in specified range
                        int64_t voxel_index = (int64_t) std::floor(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                        if (voxel_index >= (int64_t) voxel_bins) {
                            fprintf(stderr,"Out-of-bounds error for index %ld: %ld > %ld:\n", flat_idx, voxel_index, voxel_bins);
                        } else if (voxel != 0) { // Voxel not masked, and within vmin,vmax range
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
        */

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
            dz = (double)nz / ((double)nZ),
            dy = (double)ny / ((double)nY),
            dx = (double)nx / ((double)nX);

        auto [f_min, f_max] = frange;
        auto [v_min, v_max] = vrange;
        auto [z_start, y_start, x_start] = offset;
        uint64_t
            bins_length = field_bins * voxel_bins,
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

        #pragma omp parallel
        {
            uint64_t *tmp_bins = (uint64_t*) calloc(bins_length, sizeof(uint64_t));
            #pragma omp for nowait
            for (uint64_t Z = 0; Z < (uint64_t) block_size.z; Z++) {
                for (uint64_t Y = y_start; Y < y_end; Y++) {
                    for (uint64_t X = x_start; X < x_end; X++) {
                        uint64_t flat_index = (Z*nY*nX) + (Y*nX) + X;
                        auto voxel = voxels[flat_index];
                        voxel = (voxel >= v_min && voxel <= v_max) ? voxel: 0; // Mask away voxels that are not in specified range
                        int64_t voxel_index = (int64_t) std::floor(static_cast<double>(voxel_bins-1) * ((voxel - v_min) / (v_max - v_min)) );

                        // And what are the corresponding x,y,z coordinates into the field array, and field basearray index i?
                        // TODO: Sample 2x2x2 volume?
                        uint64_t
                            z = (uint64_t) std::floor((double)Z * dz),
                            y = (uint64_t) std::floor((double)Y * dy),
                            x = (uint64_t) std::floor((double)X * dx);
                        z = std::min((uint64_t)(nz-1), z); // Clamp when nZ % 2 != 0
                        uint64_t i = z*ny*nx + y*nx + x;

                        // TODO the last row of the histogram does not work, when the mask is "bright". Should be discarded.
                        if((voxel != 0) && (field[i] > 0)) { // Mask zeros in both voxels and field (TODO: should field be masked, or 0 allowed?)
                            int64_t field_index = (int64_t) std::floor(static_cast<double>(field_bins-1) * ((field[i] - f_min) / (f_max - f_min)) );

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

        auto [nZ, nY, nX] = voxels_shape;
        auto [nz, ny, nx] = field_shape;

        double
            dz = (double)nz / ((double)nZ),
            dy = (double)ny / ((double)nY),
            dx = (double)nx / ((double)nX);

        auto [f_min, f_max] = frange;
        auto [v_min, v_max] = vrange;
        auto [z_start, y_start, x_start] = offset;

        if (verbose >= 2) {
            printf("\nStarting calculation\n");
            printf("nZ, nY, nX = %ld, %ld, %ld\n", nZ, nY, nX);
            printf("nz, ny, nx = %ld, %ld, %ld\n", nz, ny, nx);
            printf("dz, dy, dx = %g, %g, %g\n", dz, dy, dx);
            printf("vrange = (%g, %g), frange = (%g, %g)\n", v_min, v_max, f_min, f_max);
            fflush(stdout);
        }

        uint64_t
            // TODO maybe make into a parameter? These values are derived from 1000_compute_histograms.py, the original was bins_info.size, which is the flat total amount of elements in the bins array.
            bins_length = nz*(voxel_bins/2)*field_bins,
            x_end = (uint64_t) std::min(x_start+block_size.z, nX),
            y_end = nY,
            z_end = nZ;

        #pragma omp parallel
        {
            uint64_t *tmp_bins = (uint64_t*) malloc(sizeof(uint64_t) * bins_length);
            #pragma omp for nowait
            for (uint64_t X = 0; X < x_end-x_start; X++) {
                for (uint64_t Y = y_start; Y < y_end; Y++) {
                    for (uint64_t Z = z_start; Z < z_end; Z++) {
                        uint64_t flat_index = (X*nY*nZ) + (Y*nZ) + Z;
                        auto voxel = voxels[flat_index];
                        voxel = (voxel >= v_min && voxel <= v_max) ? voxel: 0; // Mask away voxels that are not in specified range
                        int64_t voxel_index = (int64_t) std::floor(static_cast<double>(voxel_bins-1) * ((voxel - v_min) / (v_max - v_min)) );

                        // And what are the corresponding x,y,z coordinates into the field array, and field basearray index i?
                        // TODO: Sample 2x2x2 volume?
                        float
                            x = (float) X * (float) dx,
                            y = (float) Y * (float) dy,
                            z = (float) Z * (float) dz;
                        uint64_t i = (uint64_t) std::floor(x)*ny*nz + (uint64_t) std::floor(y)*nz + (uint64_t) std::floor(z);

                        // TODO the last row of the histogram does not work, when the mask is "bright". Should be discarded.
                        //if ((voxel >= 0) && field[i] > 0) { // Mask zeros in both voxels and field (TODO: should field be masked, or 0 allowed?)
                            field_type field_value = (field_type) (resample2x2x2(field, field_shape, {z,y,x}));
                            int64_t field_index = (int64_t) std::floor(static_cast<double>(field_bins-1) * ((field_value - f_min) / (f_max - f_min)) );


                            if (field_index < 0 || (uint64_t) field_index >= field_bins) {
                                fprintf(stderr,"field value out of bounds at X,Y,Z = %ld,%ld,%ld, x,y,z = %.1f,%.1f,%.1f:\n"
                                "\t field_value = %d (%.3f), field_index = %ld, voxel_value = %d, field[%ld] = %d\n",
                                X,Y,Z,x,y,z,
                                field_value, std::floor(resample2x2x2(field, {nx,ny,nz}, {x,y,z})), field_index, voxel, i, field[i]);
                                printf("nx,ny,nz = %ld,%ld,%ld. %ld*%ld + %ld*%ld + %ld = %ld\n",
                                    nx,ny,nz,
                                    (uint64_t) std::floor(x), ny*nz,
                                    (uint64_t) std::floor(y), nz,
                                    (uint64_t) std::floor(z),
                                    i
                                );

                                abort();
                            }


                            if((field_index >= 0) && ((uint64_t) field_index < field_bins)) // Resampling with masked voxels can give field_value < field_min
                                tmp_bins[field_index*voxel_bins + voxel_index]++;
                        //}
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

}