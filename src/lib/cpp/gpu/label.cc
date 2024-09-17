#include "label.hh"

namespace gpu {

    inline float resample2x2x2(const field_type *voxels,
            const shape_t &shape,
            const std::array<float, 3> &X) {
        auto  [Nz,Ny,Nx] = shape;
        // assert(X[0]>=0.5      && X[1]>=0.5      && X[2]>= 0.5);
        // assert(X[0]<=(Nx-0.5) && X[1]<=(Ny-0.5) && X[2]<= (Nz-0.5));

        float   Xfrac[2][3]; // {Xminus[3], Xplus[3]}
        int64_t  Xint[2][3]; // {Iminus[3], Iplus[3]}
        float   value = 0;

        for(int i = 0; i < 3; i++) {
            float Iminus, Iplus;
            Xfrac[0][i] = 1-modf(X[i]-0.5, &Iminus); // 1-{X[i]-1/2}, floor(X[i]-1/2)
            Xfrac[1][i] =   modf(X[i]+0.5, &Iplus);  // {X[i]+1/2}, floor(X[i]+1/2)

            Xint[0][i] = (int64_t) Iminus;
            Xint[1][i] = (int64_t) Iplus;
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

        for (int ijk = 0; ijk <= 7; ijk++) {
            float  weight = 1;
            int64_t IJK[3] = {0,0,0};

            for (int axis = 0; axis < 3; axis++) { // x-1/2 or x+1/2
                int pm = (ijk >> axis) & 1;
                IJK[axis] = Xint[pm][axis];
                weight   *= Xfrac[pm][axis];
            }

            auto [I,J,K] = IJK;
            if (I < 0 || J < 0 || K < 0) {
                printf("(I,J,K) = (%ld,%ld,%ld)\n", I, J, K);
                abort();
            }
            if (I >= int(Nx) || J >= int(Ny) || K >= int(Nz)) {
                printf("(I,J,K) = (%ld,%ld,%ld), (Nx,Ny,Nz) = (%ld,%ld,%ld), (X,Y,Z) = (%g,%g,%g)\n", I, J, K, Nx, Ny, Nz, X[0], X[1], X[2]);
                abort();
            }
            uint64_t voxel_index = I + J*Nx + K*Nx*Ny;
            field_type voxel = voxels[voxel_index];
            value += voxel * weight;
        }

        return value;
    }

    void material_prob_justonefieldthx(const py::array_t<voxel_type> &np_voxels,
            const py::array_t<field_type> &np_field,
            const py::array_t<prob_type> &np_prob,
            py::array_t<result_type> &np_result,
            const std::pair<voxel_type, voxel_type> &vrange,
            const std::pair<field_type, field_type> &frange,
            const std::tuple<uint64_t, uint64_t, uint64_t> &offset,
            const std::tuple<uint64_t, uint64_t, uint64_t> &ranges) {
        py::buffer_info
            voxels_info = np_voxels.request(),
            field_info  = np_field.request(),
            prob_info   = np_prob.request(),
            result_info = np_result.request();

        const voxel_type *voxels = static_cast<const voxel_type*>(voxels_info.ptr);
        const field_type *field  = static_cast<const field_type*>(field_info.ptr);
        const prob_type  *prob   = static_cast<prob_type*>(prob_info.ptr);
        result_type      *result = static_cast<result_type*>(result_info.ptr);

        const uint64_t Nfield_bins = (uint64_t) prob_info.shape[0],
                    Nvoxel_bins = (uint64_t) prob_info.shape[1];

        auto [sz, sy, sx] = offset;
        auto [Nz, Ny, Nx] = ranges;
        const auto nz = voxels_info.shape[0], ny = voxels_info.shape[1], nx = voxels_info.shape[2];
        const auto fz = field_info.shape[0],  fy = field_info.shape[1],  fx = field_info.shape[2];
        const bool field_scaled = (fz != Nz || fy != Ny || fx != Nx);
        const float
            dz = (float) fz / (float) nz,
            dy = (float) fy / (float) ny,
            dx = (float) fx / (float) nx;

        auto [v_min, v_max] = vrange;
        auto [f_min, f_max] = frange;

        printf("v_min, v_max = %d,%d\nf_min, f_max = %d,%d\n", v_min, v_max, f_min, f_max);

        #pragma acc data copyin(voxels[0:voxels_info.size], field[0:field_info.size], prob[0:prob_info.size]) copyout(result[0:result_info.size])
        {
            #pragma acc parallel loop present(result[0:result_info.size])
            for(size_t i = 0; i < (uint64_t) result_info.size; i++) {
                result[i] = 0;
            }

            #pragma acc parallel loop collapse(3) present(voxels[0:voxels_info.size], field[0:field_info.size], prob[0:prob_info.size], result[0:result_info.size])
            for (uint64_t z = sz; z < Nz; z++) {
                for (uint64_t y = sy; y < Ny; y++) {
                    for (uint64_t x = sx; x < Nx; x++) {
                        uint64_t flat_index = (z-sz)*Ny*Nx + y*Nx + x;

                        // assert(flat_index < voxels_info.size);
                        voxel_type voxel = voxels[flat_index];

                        voxel = (voxel >= v_min && voxel <= v_max) ? voxel : 0;
                        if(voxel == 0) continue;

                        // assert(voxel >= v_min && voxel <= v_max);

                        int64_t voxel_index = (int64_t) std::floor(static_cast<float>(Nvoxel_bins-1) * ((voxel - v_min) / float(v_max - v_min)) );

                        // if(!(voxel_index >= 0 && voxel_index < Nvoxel_bins)){
                        //     fprintf(stderr,"voxel = %d, voxel_index = %ld, Nvoxel_bins = %ld\n",
                        // 	   voxel,voxel_index, Nvoxel_bins);
                        //     abort();
                        // }

                        // // TODO: Allow variable voxels scale and field scale (dx = Nx/float(fx), etc.)

                        field_type field_value = 0;
                        {
                            std::array<float, 3> XYZ = {
                                (float) x      * dx,
                                (float) y      * dy,
                                (float) (z-sz) * dz
                            };
                            auto [X,Y,Z] = XYZ;
                            if (!field_scaled && X >= 0.5 && Y >= 0.5 && Z >= 0.5 &&
                                    (X+0.5) < (float)fx && (Y+0.5) < (float)fy && (Z+0.5) < (float)fz) {
                                field_value = (field_type) std::floor(resample2x2x2(
                                    field, { fz, fy, fx }, XYZ
                                ));
                            } else {
                                Z = std::min((float)(fz-1), Z); // Clamp when nz % 2 != 0
                                uint64_t i = (uint64_t) std::floor(Z)*fy*fx + (uint64_t) std::floor(Y)*fx + (uint64_t) std::floor(X);
                                field_value = field[i];
                            }
                        }

                        field_value = std::max(f_min, field_value);
                        field_value = std::min(f_max, field_value);

                        int64_t field_index = (int64_t) std::floor(static_cast<float>(Nfield_bins-1) * (float(field_value - f_min)/float(f_max - f_min)) );

                        // if(!(field_index >= 0 && field_index < Nfield_bins)){
                        //     fprintf(stderr,"field_value = %d, field_index = %ld, Nfield_bins = %ld\n",
                        // 	   field_value,field_index, Nfield_bins);
                        //     abort();
                        // }

                        result[flat_index] = (result_type) voxel_index;
                        prob_type p = prob[field_index*Nvoxel_bins + voxel_index]; //* weights[i + n_axes];
                        // if(p<0 || p>1){
                        //     fprintf(stderr,"p = %g; field_v = %d; voxel_voxel = %d\n",p, field_value,voxel);
                        //     abort();
                        // }
                        result[flat_index] = (result_type) std::round(p * std::numeric_limits<result_type>::max());
                        // result[flat_index] = p;
                    }
                }
            }
        }
    }

    #pragma GCC diagnostic ignored "-Wunused-parameter"
    void otsu(
            const np_array<uint64_t> &np_bins,
            np_array<uint64_t> &np_result,
            uint64_t step_size) {
        throw std::runtime_error("Not implemented");
    }

}