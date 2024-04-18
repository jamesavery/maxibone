#include "label.hh"

namespace cpu_par {

inline double resample2x2x2(const field_type *voxels,
            const std::tuple<uint64_t,uint64_t,uint64_t> &shape,
            const std::array<double, 3>    &X) {
    auto  [Nz,Ny,Nx] = shape;	// Eller omvendt?
    // assert(X[0]>=0.5      && X[1]>=0.5      && X[2]>= 0.5);
    // assert(X[0]<=(Nx-0.5) && X[1]<=(Ny-0.5) && X[2]<= (Nz-0.5));

    double   Xfrac[2][3];	// {Xminus[3], Xplus[3]}
    int64_t  Xint[2][3];	// {Iminus[3], Iplus[3]}
    double   value = 0;

    for(int i = 0; i < 3; i++) {
        double Iminus, Iplus;
        Xfrac[0][i] = 1-modf((double)X[i]-0.5, &Iminus); // 1-{X[i]-1/2}, floor(X[i]-1/2)
        Xfrac[1][i] =   modf((double)X[i]+0.5, &Iplus);  // {X[i]+1/2}, floor(X[i]+1/2)

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

    for(int ijk = 0; ijk <= 7; ijk++) {
        double  weight = 1;
        int64_t IJK[3] = {0,0,0};

        for(int axis = 0; axis < 3; axis++) { // x-1/2 or x+1/2
            int pm = (ijk>>axis) & 1;
            IJK[axis] = Xint[pm][axis];
            weight   *= Xfrac[pm][axis];
        }

        auto [I,J,K] = IJK;
        if(I<0 || J<0 || K<0) {
            printf("(I,J,K) = (%ld,%ld,%ld)\n",I,J,K);
            abort();
        }
        if(I>=int(Nx) || J>=int(Ny) || K>=int(Nz)) {
            printf("(I,J,K) = (%ld,%ld,%ld), (Nx,Ny,Nz) = (%ld,%ld,%ld)\n",I,J,K,Nx,Ny,Nz);
            abort();
        }
        uint64_t voxel_index = I+J*Nx+K*Nx*Ny;
        field_type voxel = voxels[voxel_index];
        value += voxel*weight;
    }

    return value;
}

void material_prob_justonefieldthx(const py::array_t<voxel_type> &np_voxels,
                   const py::array_t<field_type> &np_field,
                   const py::array_t<prob_type>  &np_prob,
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

    auto [v_min, v_max] = vrange;
    auto [f_min, f_max] = frange;

    printf("v_min, v_max = %d,%d\nf_min, f_max = %d,%d\n",v_min,v_max, f_min,f_max);

    #pragma omp parallel for
    for(size_t i = 0; i < (uint64_t) result_info.size; i++) {
        result[i] = 0;
    }

    #pragma omp parallel for collapse(3)
    for (uint64_t z = sz; z < Nz; z++) {
        for (uint64_t y = sy; y < Ny; y++) {
            for (uint64_t x = sx; x < Nx; x++) {
                uint64_t flat_index = (z-sz)*Ny*Nx + y*Nx + x;

                // assert(flat_index < voxels_info.size);
                voxel_type voxel = voxels[flat_index];

                if(voxel == 0) continue;
                voxel = std::max(v_min, voxel);
                voxel = std::min(v_max, voxel);

                // assert(voxel >= v_min && voxel <= v_max);

                int64_t voxel_index = (int64_t) floor(static_cast<double>(Nvoxel_bins-1) * ((voxel - v_min)/double(v_max - v_min)) );

                // if(!(voxel_index >= 0 && voxel_index < Nvoxel_bins)){
                //     fprintf(stderr,"voxel = %d, voxel_index = %ld, Nvoxel_bins = %ld\n",
                // 	   voxel,voxel_index, Nvoxel_bins);
                //     abort();
                // }

                // // TODO: Allow variable voxels scale and field scale (dx = Nx/float(fx), etc.)

                field_type  field_value = 0;
                {
                    double
                        dz = (double) fz / ((double) nz),
                        dy = (double) fy / ((double) ny),
                        dx = (double) fx / ((double) nx);

                    std::array<double, 3> XYZ = {
                        (double) x      * dx,
                        (double) y      * dy,
                        (double) (z-sz) * dz
                    };
                    auto [X,Y,Z] = XYZ;
                    if (X>=0.5 && Y>=0.5 && Z>=0.5 &&
                            (X+0.5)<(double)nx && (Y+0.5)<(double)ny && (Z+0.5)<(double)nz) {
                        field_value = (field_type) round(resample2x2x2(
                            field,
                            { field_info.shape[0], field_info.shape[1], field_info.shape[2] },
                            XYZ
                        ));
                    } else {
                        uint64_t i = (uint64_t) floor(Z)*ny*nx + (uint64_t) floor(Y)*nx + (uint64_t) floor(X);
                        field_value = field[i];
                    }
                }


                field_value = std::max(f_min, field_value);
                field_value = std::min(f_max, field_value);

                int64_t field_index = (int64_t) floor(static_cast<double>(Nfield_bins-1) * (double(field_value - f_min)/double(f_max - f_min)) );

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
                result[flat_index] = (result_type) round(p * std::numeric_limits<result_type>::max());
                // result[flat_index] = p;
            }
        }
    }
    // fprintf(stderr,"maxint = %d\n",std::numeric_limits<result_type>::max());
    // result_type r_min = 0, r_max = 0;
    // for(size_t i=0;i<result_info.size;i++){ r_min = std::min(r_min,result[i]); r_max = std::max(r_max,result[i]); }
    // fprintf(stderr,"result min,max = %d,%d\n",r_min,r_max);
}

void otsu(
        const np_array<uint64_t> np_bins,
        np_array<uint64_t> np_result,
        const uint64_t step_size) {
    py::buffer_info
        bins_info = np_bins.request(),
        result_info = np_result.request();
    // https://vincmazet.github.io/bip/segmentation/histogram.html

    uint64_t N_rows = bins_info.shape[0], N_cols = bins_info.shape[1];
    uint64_t N_threshes = N_cols / step_size;

    const uint64_t *bins = static_cast<const uint64_t*>(bins_info.ptr);
    uint64_t *result = static_cast<uint64_t*>(result_info.ptr);

    #pragma omp parallel for
    for (uint64_t i = 0; i < N_rows; i++) {
        const uint64_t *row = bins + (i*N_cols);

        uint64_t *guesses = (uint64_t*) malloc(sizeof(uint64_t) * N_threshes);
        for (uint64_t guess = 0; guess < N_threshes; guess++) {
            uint64_t w0 = 0, w1 = 0, th = guess * step_size;

            // w0 = row[:th].sum()
            for (uint64_t j = 0; j < th; j++) {
                w0 += row[j];
            }

            // w1 = row[th:].sum()
            for (uint64_t j = th; j < N_cols; j++) {
                w1 += row[j];
            }

            //if w0 <= 0 or w1 <= 0:
            //    return np.inf
            if (w0 == 0 || w1 == 0) {
                guesses[guess] = (uint64_t) -1;
            } else {
                double fw0 = 1.0 / (double) w0, fw1 = 1.0 / (double) w1;
                uint64_t tm0 = 0, tm1 = 0;

                // m0 = (1/w0) * (np.arange(th)*row[:th]).sum()
                for (uint64_t j = 0; j < th; j++) {
                    tm0 += j * row[j];
                }
                double m0 = fw0 * ((double)tm0);

                // m1 = (1/w1) * (np.arange(th,row.shape[0])*row[th:]).sum()
                for (uint64_t j = th; j < N_cols; j++) {
                    tm1 += j * row[j];
                }
                double m1 = fw1 * ((double)tm1);

                double s0 = 0.0, s1 = 0.0;
                // s0 = (1/w0) * (((np.arange(th)-m0)**2)*row[:th]).sum()
                for (uint64_t j = 0; j < th; j++) {
                    double im0 = (double)j - m0;
                    s0 += (im0*im0) * (double)row[j];
                }
                s0 *= fw0;

                // s1 = (1/w1) * (((np.arange(row.shape[0]-th)-m1)**2)*row[th:]).sum()
                for (uint64_t j = th; j < N_cols; j++) {
                    double im1 = (double)j - m1;
                    s1 += (im1*im1) * (double)row[j];
                }
                s1 *= fw1;

                // return w0*s0 + w1*s1
                guesses[guess] = (uint64_t) floor((double)w0*s0 + (double)w1*s1);
            }
        }

        uint64_t min_idx = 0;
        for (uint64_t guess = 1; guess < N_threshes; guess++) {
            min_idx = guesses[guess] < guesses[min_idx] ? guess : min_idx;
        }
        free(guesses);

        result[i] = min_idx * step_size;
    }
}

}