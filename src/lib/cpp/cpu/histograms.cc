#include <histograms.hh>

namespace cpu_par {

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
                        int64_t voxel_index = (int64_t) round(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                        if (voxel_index >= (int64_t) voxel_bins) {
                            fprintf(stderr,"Out-of-bounds error for index %lld: %lld > %lld:\n", flat_idx, voxel_index, voxel_bins);
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
                for (uint64_t z = 0; z < z_end-z_start; z++) {
                    for (uint64_t x = x_start; x < x_end; x++) {
                        uint64_t flat_idx = z*Ny*Nx + y*Nx + x;
                        auto voxel = voxels[flat_idx];
                        voxel = (voxel >= vmin && voxel <= vmax) ? voxel: 0; // Mask away voxels that are not in specified range
                        int64_t voxel_index = (int64_t) round(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                        if (voxel_index >= (int64_t) voxel_bins) {
                            fprintf(stderr,"Out-of-bounds error for index %lld: %lld > %lld:\n", flat_idx, voxel_index, voxel_bins);
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
                        int64_t voxel_index = (int64_t) round(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                        if (voxel_index >= (int64_t) voxel_bins) {
                            fprintf(stderr,"Out-of-bounds error for index %lld: %lld > %lld:\n", flat_idx, voxel_index, voxel_bins);
                        } else if (voxel != 0) { // Voxel not masked, and within vmin,vmax range
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

                    uint64_t r = (uint64_t) floor(sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy)));

                    // Read and compute
                    #pragma omp simd
                    for (uint64_t z = 0; z < z_end-z_start; z++) {
                        uint64_t flat_idx = z*Ny*Nx + y*Nx + x;
                        auto voxel = voxels[flat_idx];
                        voxel = (voxel >= vmin && voxel <= vmax) ? voxel: 0; // Mask away voxels that are not in specified range
                        int64_t voxel_index = (int64_t) round(static_cast<double>(voxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );

                        if (voxel_index >= (int64_t) voxel_bins) {
                            fprintf(stderr,"Out-of-bounds error for index %lld: %lld > %lld:\n", flat_idx, voxel_index, voxel_bins);
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
            // TODO maybe make into a parameter? These values are derived from 1000_compute_histograms.py, the original was bins_info.size, which is the flat total amount of elements in the bins array.
            bins_length = nz*(voxel_bins/2)*field_bins,
            z_end = (uint64_t) std::min(z_start+block_size.z, nZ),
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
                        int64_t voxel_index = (int64_t) floor(static_cast<double>(voxel_bins-1) * ((voxel - v_min)/(v_max - v_min)) );

                        // And what are the corresponding x,y,z coordinates into the field array, and field basearray index i?
                        // TODO: Sample 2x2x2 volume?
                        uint64_t
                            z = (uint64_t) floor((double)Z*dz),
                            y = (uint64_t) floor((double)Y*dy),
                            x = (uint64_t) floor((double)X*dx);
                        uint64_t i = z*ny*nx + y*nx + x;

                        // TODO the last row of the histogram does not work, when the mask is "bright". Should be discarded.
                        if((voxel != 0) && (field[i] > 0)) { // Mask zeros in both voxels and field (TODO: should field be masked, or 0 allowed?)
                            int64_t field_index = (int64_t) floor(static_cast<double>(field_bins-1) * ((field[i] - f_min)/(f_max - f_min)) );

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

    void field_histogram_resample(const np_array<voxel_type> np_voxels,
                        const np_array<field_type> np_field,
                        const array<uint64_t,3> offset,
                        const array<uint64_t,3> voxels_shape,
                        const array<uint64_t,3> field_shape,
                        const uint64_t block_size,
                        np_array<uint64_t> &np_bins,
                        const array<double, 2> vrange,
                        const array<double, 2> frange) {
        py::buffer_info
            voxels_info = np_voxels.request(),
            field_info = np_field.request(),
            bins_info = np_bins.request();

        const uint64_t
            bins_length  = bins_info.size,
            field_bins   = bins_info.shape[0],
            voxel_bins   = bins_info.shape[1];

        auto [nX, nY, nZ] = voxels_shape;
        auto [nx, ny, nz] = field_shape;

        double dx = nx/((double)nX), dy = ny/((double)nY), dz = nz/((double)nZ);

        const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);
        const field_type *field  = static_cast<field_type*>(field_info.ptr);
        uint64_t         *bins   = static_cast<uint64_t*>(bins_info.ptr);

        auto [f_min, f_max] = frange;
        auto [v_min, v_max] = vrange;
        auto [z_start, y_start, x_start] = offset;
        uint64_t
            x_end = min(x_start+block_size, nX),
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
                        int64_t voxel_index = floor(static_cast<double>(voxel_bins-1) * ((voxel - v_min)/(v_max - v_min)) );

                        // And what are the corresponding x,y,z coordinates into the field array, and field basearray index i?
                        // TODO: Sample 2x2x2 volume?
                        float    x = X*dx, y = Y*dy, z = Z*dz;
                uint64_t i = floor(x)*ny*nz + floor(y)*nz + floor(z);



                        // TODO the last row of the histogram does not work, when the mask is "bright". Should be discarded.
                        if(VALID_VOXEL(voxel) && field[i]>0) { // Mask zeros in both voxels and field (TODO: should field be masked, or 0 allowed?)
                field_type field_value = round(resample2x2x2(field,{nx,ny,nz},{x,y,z}));
                int64_t field_index = floor(static_cast<double>(field_bins-1) * ((field_value - f_min)/(f_max - f_min)) );


                if(field_index < 0 || field_index >= field_bins){
                fprintf(stderr,"field value out of bounds at X,Y,Z = %ld,%ld,%ld, x,y,z = %.1f,%.1f,%.1f:\n"
                    "\t field_value = %d (%.3f), field_index = %ld, voxel_value = %d, field[%ld] = %d\n",
                    X,Y,Z,x,y,z,
                    field_value, round(resample2x2x2(field,{nx,ny,nz},{x,y,z})), field_index, voxel,i,field[i]);
                printf("nx,ny,nz = %ld,%ld,%ld. %ld*%ld + %ld*%ld + %ld = %ld\n",
                    nx,ny,nz,
                    int(floor(x)),ny*nz,
                    int(floor(y)),nz,
                    int(floor(z)),
                    i
                    );

                abort();
                }


                if((field_index >= 0) && (field_index < field_bins)) // Resampling with masked voxels can give field_value < field_min
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

}