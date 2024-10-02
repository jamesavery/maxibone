// TODO: Coordinates are named X,Y,Z in c++, but Z,Y,X in python. Homogenize to X,Y,Z!
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <math.h>

#include "boilerplate.hh"
#include "geometry.hh"

namespace cpu_seq {

    std::array<real_t, 3> center_of_mass(const input_ndarray<mask_type> &mask) {
        UNPACK_NUMPY(mask);

        uint64_t total_mass = 0, cmz = 0, cmy = 0, cmx = 0;

        BLOCK_BEGIN(mask, reduction(+:total_mass,cmz,cmy,cmx)); {

            mask_type m = mask_buffer[flat_index];

            total_mass += m;
            cmz += m * z;
            cmy += m * y;
            cmx += m * x;

        } BLOCK_END();

        real_t
            rcmz = real_t(cmz) / real_t(total_mass),
            rcmy = real_t(cmy) / real_t(total_mass),
            rcmx = real_t(cmx) / real_t(total_mass);

        return std::array<real_t, 3>{ rcmz, rcmy, rcmx };
    }

    void center_of_masses(const input_ndarray<uint64_t> &mask, output_ndarray<real_t> &output) {
        UNPACK_NUMPY(mask);
        UNPACK_NUMPY(output);

        const uint64_t *mask_data = mask.data;
        ssize_t n_masks = output.shape[0];
        real_t *output_data = output.data;

        uint64_t
            *total_masses = (uint64_t *) malloc(n_masks * sizeof(uint64_t)),
            *cmzs = (uint64_t *) malloc(n_masks * sizeof(uint64_t)),
            *cmys = (uint64_t *) malloc(n_masks * sizeof(uint64_t)),
            *cmxs = (uint64_t *) malloc(n_masks * sizeof(uint64_t));

        #pragma omp parallel for
        for (int64_t z = 0; z < mask_Nz; z++) {
            for (int64_t y = 0; y < mask_Ny; y++) {
                for (int64_t x = 0; x < mask_Nx; x++) {
                    int64_t flat_index = z*mask_Ny*mask_Nx + y*mask_Nx + x;
                    uint64_t m = mask_data[flat_index];

                    if (m) { // We don't need to compute the background.
                        #pragma omp atomic
                        total_masses[m] += 1;
                        #pragma omp atomic
                        cmzs[m] += z;
                        #pragma omp atomic
                        cmys[m] += y;
                        #pragma omp atomic
                        cmxs[m] += x;
                    }
                }
            }
        }

        #pragma omp parallel for
        for (int64_t m = 1; m < n_masks; m++) { // Skip the background.
            output_data[m*3 + 0] = (real_t) cmzs[m] / (real_t) total_masses[m];
            output_data[m*3 + 1] = (real_t) cmys[m] / (real_t) total_masses[m];
            output_data[m*3 + 2] = (real_t) cmxs[m] / (real_t) total_masses[m];
        }

        free(total_masses);
        free(cmzs);
        free(cmys);
        free(cmxs);
    }

    void compute_front_mask(const input_ndarray<mask_type> solid_implant,
            const float voxel_size,
            const matrix4x4 &Muvw,
            std::array<float, 6> bbox,
            output_ndarray<mask_type> front_mask) {
        const auto [U_min, U_max, V_min, V_max, W_min, W_max] = bbox;
        UNPACK_NUMPY(solid_implant)

        // TODO move the typedefs here, rather than having them globally in datatypes.hh
        BLOCK_BEGIN_WITH_OUTPUT(solid_implant, front_mask, ) {

            std::array<real_t, 4> Xs = {
                real_t(z) * voxel_size,
                real_t(y) * voxel_size,
                real_t(x) * voxel_size,
                1 };
            mask_type mask_value = solid_implant_buffer[flat_index];

            if (mask_value) {
                front_mask_buffer[flat_index] = 0;
            } else {
                auto [U,V,W,c] = hom_transform(Xs, Muvw);
                front_mask_buffer[flat_index] = W > W_min;
            }

        BLOCK_END_WITH_OUTPUT() }
    }

    void compute_front_back_masks(const mask_type *mask, const shape_t &shape, const float voxel_size, const float *E, const float *cm, const float *cp, const float *UVWp, mask_type *front_mask, mask_type *back_mask, mask_type *implant_shell_mask, mask_type *solid_implant) {
        auto [nz, ny, nx] = shape;

        // Python code:
        // implant_zyxs = np.std::array(np.nonzero(implant)).T - cm // Implant points in z,y,x-coordinates (relative to upper-left-left corner, in {scale}x voxel units)
        // implant_uvws = implant_zyxs @ E                          // Implant points in u,v,w-coordinates (relative to origin cm, in {scale}x voxel units)
        // w0  = implant_uvws[:,2].min()                            // In {scale}x voxel units
        // w0v = np.std::array([0,0,w0])                            // w-shift to get to center of implant back-plane
        float w0v[3] = {0, 0, 0};
        #pragma omp parallel for collapse(3) reduction(min:w0v[:3])
        for (int64_t z = 0; z < nz; z++) {
            for (int64_t y = 0; y < ny; y++) {
                for (int64_t x = 0; x < nx; x++) {
                    int64_t flat_index = z*ny*nx + y*nx + x;
                    if (mask[flat_index]) {
                        float zyxs[3] = {
                            float(z) - cm[0],
                            float(y) - cm[1],
                            float(x) - cm[2]
                        };
                        float uvws[3] = {
                            zyxs[0] * E[0] + zyxs[1] * E[3] + zyxs[2] * E[6],
                            zyxs[0] * E[1] + zyxs[1] * E[4] + zyxs[2] * E[7],
                            zyxs[0] * E[2] + zyxs[1] * E[5] + zyxs[2] * E[8]
                        };
                        w0v[2] = std::min(w0v[2], uvws[2]);
                    }
                }
            }
        }

        // Python code:
        // shapes:  E: (3, 3), cm: (3,), w0v: (3,), cp: (3,), UVWp: (3, 3)
        // zyxs = coordinate_image(implant.shape)
        // uvws = (zyxs - cm) @ E
        // UVWs = (uvws - w0v) * voxel_size
        // UVWps = (UVWs - cp) @ UVWp
        // Us,Vs,Ws = UVWs[...,0], UVWs[...,1], UVWs[...,2]
        // Ups,Vps,Wps = UVWps[...,0], UVWps[...,1], UVWps[...,2]
        // thetas, rs = np.arctan2(Vps,Wps), np.sqrt(Vps**2+Wps**2)
        // rmaxs = (rs*(implant==True)).reshape(nz,-1).max(axis=1)[:,NA,NA]
        // implant_shell_mask = implant&(rs >= 0.7*rmaxs)
        // solid_implant = (implant | (rs < 0.7*rmaxs) & (Ws >= 0))
        // back_mask  = (Ws<0)
        // front_mask = largest_cc_of((Ws>50)&(~solid_implant))
        float *rs, *Ws;
        #pragma omp parallel private(rs, Ws) shared(mask, front_mask, back_mask, implant_shell_mask, solid_implant)
        {
            rs = (float *) malloc(ny * nx * sizeof(float));
            Ws = (float *) malloc(ny * nx * sizeof(float));

            #pragma omp for schedule(static)
            for (int64_t z = 0; z < nz; z++) {
                float rmax = 0.0f;
                for (int64_t y = 0; y < ny; y++) {
                    for (int64_t x = 0; x < nx; x++) {
                        int64_t flat_index = z*ny*nx + y*nx + x;
                        float uvw[3] = {
                            (float) (((float)z)-cm[0]) * E[0] + (((float)y)-cm[1]) * E[3] + (((float)x)-cm[2]) * E[6],
                            (float) (((float)z)-cm[0]) * E[1] + (((float)y)-cm[1]) * E[4] + (((float)x)-cm[2]) * E[7],
                            (float) (((float)z)-cm[0]) * E[2] + (((float)y)-cm[1]) * E[5] + (((float)x)-cm[2]) * E[8]
                        };
                        float UVW[3] = {
                            (uvw[0] - w0v[0]) * voxel_size,
                            (uvw[1] - w0v[1]) * voxel_size,
                            (uvw[2] - w0v[2]) * voxel_size
                        };
                        float UVWps[3] = {
                            (UVW[0] - cp[0]) * UVWp[0] + (UVW[1] - cp[1]) * UVWp[3] + (UVW[2] - cp[2]) * UVWp[6],
                            (UVW[0] - cp[0]) * UVWp[1] + (UVW[1] - cp[1]) * UVWp[4] + (UVW[2] - cp[2]) * UVWp[7],
                            (UVW[0] - cp[0]) * UVWp[2] + (UVW[1] - cp[1]) * UVWp[5] + (UVW[2] - cp[2]) * UVWp[8]
                        };
                        float
                            //U = UVW[0],
                            //V = UVW[1],
                            W = UVW[2],
                            //Up = UVWps[0],
                            Vp = UVWps[1],
                            Wp = UVWps[2],
                            //theta = std::atan2(Vp, Wp),
                            r = std::sqrt(Vp*Vp + Wp*Wp);

                        rmax = std::max(rmax, r * mask[flat_index]);
                        rs[y*nx + x] = r;
                        Ws[y*nx + x] = W;
                    }
                }
                for (int64_t y = 0; y < ny; y++) {
                    for (int64_t x = 0; x < nx; x++) {
                        int64_t flat_index = z*ny*nx + y*nx + x;
                        float r = rs[y*nx + x];
                        float W = Ws[y*nx + x];
                        implant_shell_mask[flat_index] = mask[flat_index] & (r >= 0.7 * rmax);
                        solid_implant[flat_index] = mask[flat_index] | (r < 0.7 * rmax && W >= 0);
                        back_mask[flat_index] = W < 0;
                        front_mask[flat_index] = (W > 50) & !solid_implant[flat_index];
                    }
                }
            }

            free(rs);
            free(Ws);
        }
    }

    void cylinder_projection(const input_ndarray<float> edt,
            const input_ndarray<uint8_t> C,
            float voxel_size,
            float d_min, float d_max,
            float theta_min, float theta_max,
            std::array<float, 6> bbox,
            const matrix4x4 &Muvw,
            output_ndarray<float> image,
            output_ndarray<int64_t> count,
            const int verbose) {
        UNPACK_NUMPY(C);
        UNPACK_NUMPY(edt);

        ssize_t n_theta = image.shape[0], n_U = image.shape[1];

        const auto& [U_min, U_max, V_min, V_max, W_min, W_max] = bbox;

        real_t
            edz = real_t(edt_Nz) / real_t(C_Nz),
            edy = real_t(edt_Ny) / real_t(C_Ny),
            edx = real_t(edt_Nx) / real_t(C_Nx);

        if (verbose >= 2) {
            printf("Segmenting from %g to %g micrometers distance of implant.\n", d_min, d_max);
            printf("Bounding box is [U_min,U_max,V_min,V_max,W_min,W_max] = [[%g,%g],[%g,%g],[%g,%g]]\n",
                U_min, U_max, V_min, V_max, W_min, W_max);
            printf("EDT field is (%f,%f,%f)\n", edx, edy, edz);
        }

        real_t th_min = 1234, th_max = -1234;
        ssize_t n_shell = 0;
        ssize_t n_shell_bbox = 0;

        ssize_t block_height = 64;

        {
            float   *image_d = image.data;
            int64_t *count_d = count.data;

            for (ssize_t block_start = 0, edt_block_start = 0; block_start < C_length; block_start += block_height*C_Ny*C_Nz, edt_block_start += block_height*edt_Ny*edt_Nz) {
                const uint8_t *C_buffer = C.data + block_start;
                const float  *edt_block = edt.data + std::max(block_start - edt_Ny*edt_Nz, 0L);

                ssize_t  this_block_length = std::min(block_height*C_Ny*C_Nz, C_length-block_start);
                ssize_t  this_edt_length   = std::min((block_height+2)*edt_Ny*edt_Nz, edt_length-block_start);

                //#pragma acc parallel loop copy(C_buffer[:this_block_length], image_d[:n_theta*n_U], count_d[:n_theta*n_U], bbox[:6], Muvw[:16], edt_block[:this_edt_length]) reduction(+:n_shell,n_shell_bbox)
                //#pragma omp parallel for reduction(+:n_shell,n_shell_bbox)
                for (int64_t k = 0; k < this_block_length; k++) {
                    const int64_t flat_idx = block_start + k;
                    const int64_t
                        X = (flat_idx / (C_Ny*C_Nz)),
                        Y = (flat_idx / C_Nz) % C_Ny,
                        Z = flat_idx  % C_Nz; // Integer indices: Cs[c,X,Y,Z]

                    // Index into local block
                    const int64_t
                        Xl = (k  / (C_Ny*C_Nz)),
                        Yl = (k / C_Nz) % C_Ny,
                        Zl = k  % C_Nz;

                    // Index into local edt block. Note EDT has 1-slice padding top+bottom
                    const float
                        x = float(Xl+1)*edx,
                        y = float(Yl)*edy,
                        z = float(Zl)*edy;

                    if (x > float(block_height)) {
                        printf("Block number k=%ld.\nX,Y,Z=%ld,%ld,%ld\nXl,Yl,Zl=%ld,%ld,%ld\nx,y,z=%.2f, %.2f, %.2f\n",k,X,Y,Z,Xl,Yl,Zl,x,y,z);
                        abort();
                    }

                    // ****** MEAT OF THE IMPLEMENTATION IS HERE ******
                    real_t distance = resample2x2x2<float>(edt_block, { this_edt_length/(edt_Ny*edt_Nz), edt_Ny, edt_Nz }, {x,y,z});

                    if (distance > d_min && distance <= d_max) { // TODO: and W>w_min
                        std::array<real_t,4> Xs = { real_t(X)*voxel_size, real_t(Y)*voxel_size, real_t(Z)*voxel_size, 1 };
                        auto [U,V,W,c] = hom_transform(Xs, Muvw);
                        n_shell++;

                        if (verbose >= 2) {
                            printf("distance = %.1f, U,V,W = %.2f,%.2f,%.2f\n", distance, U, V, W);
                        }
                        if (in_bbox({ {U,V,W} }, bbox)) {
                            real_t theta = std::atan2(V, W);

                            if (theta >= theta_min && theta <= theta_max) {
                                n_shell_bbox++;

                                ssize_t theta_i = ssize_t(std::floor( (theta-theta_min) * real_t(n_theta-1)/(theta_max-theta_min) ));
                                ssize_t U_i     = ssize_t(std::floor( (U    -    U_min) * real_t(n_U    -1)/(    U_max-    U_min) ));

                                real_t p = real_t(C_buffer[k]) / 255.f;

                                assert(theta >= theta_min);
                                assert(theta <= theta_max);
                                assert(U >= U_min);
                                assert(U <= U_max);
                                assert(theta_i >= 0);
                                assert(theta_i < n_theta);
                                assert(U_i >= 0);
                                assert(U_i < n_U);

                                if (p > 0) {
                                    th_min = std::min(theta, th_min);
                                    th_max = std::max(theta, th_max);

                                    //atomic_statement()
                                    image_d[theta_i*n_U + U_i] += p;

                                    //atomic_statement()
                                    count_d[theta_i*n_U + U_i] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (verbose >= 2) {
            printf("n_shell = %ld, n_shell_bbox = %ld\n", n_shell, n_shell_bbox);
            printf("theta_min, theta_max = %.2f,%.2f\n", theta_min, theta_max);
            printf("th_min,       th_max = %.2f,%.2f\n", th_min, th_max);
        }
    }

    void fill_implant_mask(const input_ndarray<mask_type> mask,
            int64_t offset,
            float voxel_size,
            const std::array<float, 6> &bbox,
            float r_fraction,
            const matrix4x4 &Muvw,
            const input_ndarray<real_t> thetas,
            const input_ndarray<float> rsqr_maxs,
            output_ndarray<mask_type> solid_implant_mask,
            output_ndarray<float> profile) {
        UNPACK_NUMPY(mask)

        const real_t
            *thetas_d = thetas.data,
            theta_min = thetas_d[0],
            theta_max = thetas_d[1],
            theta_center = (theta_max + theta_min) / 2;
        ssize_t n_segments = rsqr_maxs.shape[0];
        const auto [U_min, U_max, V_min, V_max, W_min, W_max] = bbox;
        const float *rsqr_maxs_d = rsqr_maxs.data;
        float *profile_d = profile.data;

        #pragma omp parallel for collapse(3) reduction(+:profile_d[:n_segments])
        for (int64_t z = 0; z < mask_Nz; z++) {
            for (int64_t y = 0; y < mask_Ny; y++) {
                for (int64_t x = 0; x < mask_Nx; x++) {
                    int64_t z_offset = offset / (mask_Ny * mask_Nx);
                    std::array<real_t, 4> Xs = {
                        real_t(z + z_offset) * voxel_size,
                        real_t(y) * voxel_size,
                        real_t(x) * voxel_size,
                        1 };
                    int64_t flat_index = z*mask_Ny*mask_Nx + y*mask_Nx + x;
                    mask_type mask_value = mask.data[flat_index];

                    // Second pass does the actual work
                    auto [U,V,W,c] = hom_transform(Xs, Muvw);
                    float r_sqr = V*V + W*W;
                    float theta = std::atan2(V, W);
                    int U_i = int(std::floor((U - U_min) * real_t(n_segments-1) / (U_max - U_min)));

                    bool solid_mask_value = false;
                    if (U_i >= 0 && U_i < n_segments && W >= W_min) { // TODO: Full bounding box check?
                        solid_mask_value = mask_value | (r_sqr <= r_fraction * rsqr_maxs_d[U_i]);

                        if (theta >= theta_min && theta <= theta_center && r_sqr <= rsqr_maxs_d[U_i]) {
                            profile_d[U_i] += solid_mask_value;
                        }
                    }

                    solid_implant_mask.data[offset + flat_index] = solid_mask_value;
                }
            }
        }
    }

    void fill_implant_mask_pre(const input_ndarray<mask_type> mask,
            int64_t offset,
            float voxel_size,
            const std::array<float, 6> &bbox,
            const matrix4x4 &Muvw,
            output_ndarray<real_t> thetas,
            output_ndarray<float> rsqr_maxs) {
        UNPACK_NUMPY(mask);

        real_t
            *thetas_d = thetas.data,
            theta_min = thetas_d[0],
            theta_max = thetas_d[1];

        if (offset == 0) {
            thetas_d[0] = real_t(M_PI);
            thetas_d[1] = real_t(-M_PI);
        }
        ssize_t n_segments = rsqr_maxs.shape[0];
        const auto [U_min, U_max, V_min, V_max, W_min, W_max] = bbox;
        float *rsqr_maxs_d = rsqr_maxs.data;

        #pragma omp parallel for collapse(3) reduction(max:rsqr_maxs_d[:n_segments], theta_max) reduction(min:theta_min)
        for (int64_t z = 0; z < mask_Nz; z++) {
            for (int64_t y = 0; y < mask_Ny; y++) {
                for (int64_t x = 0; x < mask_Nx; x++) {
                    int64_t z_offset = offset / (mask_Ny * mask_Nx);
                    mask_type mask_value = mask.data[z*mask_Ny*mask_Nx + y*mask_Nx + x];
                    std::array<real_t, 4> Xs = {
                        real_t(z + z_offset) * voxel_size,
                        real_t(y) * voxel_size,
                        real_t(x) * voxel_size,
                        1 };

                    if (mask_value) {
                        auto [U,V,W,c] = hom_transform(Xs, Muvw);

                        real_t r_sqr = V*V + W*W;
                        real_t theta = std::atan2(V, W);

                        int U_i = int(std::floor((U - U_min) * real_t(n_segments-1) / (U_max - U_min)));

                        // if (U_i >= 0 && U_i < n_segments) {
                        if ( in_bbox({{U, V, W}}, bbox) ) {
                            rsqr_maxs_d[U_i] = std::max(rsqr_maxs_d[U_i], float(r_sqr));
                            theta_min = std::min(theta_min, theta);
                            theta_max = std::max(theta_max, theta);
                        } else {
                            // Otherwise we've calculated it wrong!
                        }
                    }
                }
            }
        }
    }

    std::array<real_t,9> inertia_matrix(const input_ndarray<mask_type> &mask, const std::array<real_t, 3> &cm, const int verbose) {
        UNPACK_NUMPY(mask);

        real_t
            Izz = 0, Izy = 0, Izx = 0,
                     Iyy = 0, Iyx = 0,
                              Ixx = 0;

        if (verbose >= 2) {
            print_timestamp("inertia_matrix_serial start");
        }

        BLOCK_BEGIN(mask, reduction(+:Izz, Iyy, Ixx) reduction(+:Izy,Izx,Iyx)) {

            real_t m = mask_buffer[flat_index];

            // m guards this, and then branches are removed
            //if (m != 0)
            real_t
                Z = ((real_t) z) - cm[0],
                Y = ((real_t) y) - cm[1],
                X = ((real_t) x) - cm[2];

            Izz += m * (Y*Y + X*X);
            Iyy += m * (Z*Z + X*X);
            Ixx += m * (Z*Z + Y*Y);
            Izy -= m * Z*Y;
            Izx -= m * Z*X;
            Iyx -= m * Y*X;

        } BLOCK_END();

        if (verbose >= 2) {
            print_timestamp("inertia_matrix_serial end");
        }

        return std::array<real_t,9> {
            Izz, Izy, Izx,
            Izy, Iyy, Iyx,
            Izx, Iyx, Ixx
        };
    }

    void inertia_matrices(const input_ndarray<uint64_t> &mask, const input_ndarray<real_t> &cms, output_ndarray<real_t> &output) {
        UNPACK_NUMPY(mask);
        UNPACK_NUMPY(cms);
        UNPACK_NUMPY(output);

        const uint64_t *mask_data = mask.data;
        const real_t *cms_data = cms.data;
        ssize_t n_masks = output.shape[0];
        real_t *output_data = output.data;

        real_t
            *Izzs = (real_t *) malloc(n_masks * sizeof(real_t)),
            *Izys = (real_t *) malloc(n_masks * sizeof(real_t)),
            *Izxs = (real_t *) malloc(n_masks * sizeof(real_t)),
            *Iyys = (real_t *) malloc(n_masks * sizeof(real_t)),
            *Iyxs = (real_t *) malloc(n_masks * sizeof(real_t)),
            *Ixxs = (real_t *) malloc(n_masks * sizeof(real_t));

        #pragma omp parallel for
        for (int64_t z = 0; z < mask_Nz; z++) {
            for (int64_t y = 0; y < mask_Ny; y++) {
                for (int64_t x = 0; x < mask_Nx; x++) {
                    int64_t flat_index = z*mask_Ny*mask_Nx + y*mask_Nx + x;
                    uint64_t m = mask_data[flat_index];
                    if (m) { // We don't need to compute the background.
                        real_t
                            Z = (real_t) z - cms_data[m*3 + 0],
                            Y = (real_t) y - cms_data[m*3 + 1],
                            X = (real_t) x - cms_data[m*3 + 2];

                        #pragma omp atomic
                        Izzs[m] += (Y*Y + X*X);
                        #pragma omp atomic
                        Iyys[m] += (Z*Z + X*X);
                        #pragma omp atomic
                        Ixxs[m] += (Z*Z + Y*Y);
                        #pragma omp atomic
                        Izys[m] -= Z*Y;
                        #pragma omp atomic
                        Izxs[m] -= Z*X;
                        #pragma omp atomic
                        Iyxs[m] -= Y*X;
                    }
                }
            }
        }

        #pragma omp parallel for
        for (int64_t m = 1; m < n_masks; m++) { // Skip the background.
            output_data[m*9 + 0] = Izzs[m];
            output_data[m*9 + 1] = Izys[m];
            output_data[m*9 + 2] = Izxs[m];
            output_data[m*9 + 3] = Izys[m];
            output_data[m*9 + 4] = Iyys[m];
            output_data[m*9 + 5] = Iyxs[m];
            output_data[m*9 + 6] = Izxs[m];
            output_data[m*9 + 7] = Iyxs[m];
            output_data[m*9 + 8] = Ixxs[m];
        }

        free(Izzs);
        free(Izys);
        free(Izxs);
        free(Iyys);
        free(Iyxs);
        free(Ixxs);
    }


    void integrate_axes(const input_ndarray<mask_type> &mask,
            const std::array<real_t, 3> &x0,
            const std::array<real_t, 3> &v_axis,
            const std::array<real_t, 3> &w_axis,
            const real_t v_min, const real_t w_min,
            output_ndarray<uint64_t> output) {
        UNPACK_NUMPY(mask);
        ssize_t Nv = output.shape[0], Nw = output.shape[1];
        uint64_t *output_data = output.data;

        // TODO: Check v_axis & w_axis projections to certify bounds and get rid of runtime check
        #pragma acc data copy(output_data[:Nv*Nw]) copyin(x0, v_axis, w_axis, v_min, w_min)
        {
        BLOCK_BEGIN(mask, ) {

            mask_type voxel = mask_buffer[flat_index];
            if (voxel != 0) {
                real_t xs[3] = {
                    real_t(x) - x0[0],
                    real_t(y) - x0[1],
                    real_t(z) - x0[2]
                };

                real_t
                    v = DOT(xs, v_axis),
                    w = DOT(xs, w_axis);
                int64_t
                    i_v = int64_t(round(v - v_min)),
                    j_w = int64_t(round(w - w_min));

                if (i_v >= 0 && j_w >= 0 && i_v < Nv && j_w < Nw) {
                    ATOMIC()
                    output_data[i_v*Nw + j_w] += voxel;
                }
            }

        BLOCK_END() }
        }
    }

    void outside_ellipsoid(const input_ndarray<uint64_t> &voxels, const input_ndarray<real_t> &cms, const input_ndarray<real_t> &abc, output_ndarray<uint64_t> &errors) {
        UNPACK_NUMPY(voxels);
        UNPACK_NUMPY(cms);
        UNPACK_NUMPY(abc);
        UNPACK_NUMPY(errors);

        const uint64_t *voxels_data = voxels.data;
        const real_t *cms_data = cms.data;
        const real_t *abc_data = abc.data;
        uint64_t *errors_data = errors.data;

        #pragma omp parallel for
        for (int64_t z = 0; z < voxels_Nz; z++) {
            for (int64_t y = 0; y < voxels_Ny; y++) {
                for (int64_t x = 0; x < voxels_Nx; x++) {
                    int64_t flat_index = z*voxels_Ny*voxels_Nx + y*voxels_Nx + x;
                    uint64_t m = voxels_data[flat_index];

                    real_t
                        Z = std::abs((real_t) z - cms_data[m*3 + 0]),
                        Y = std::abs((real_t) y - cms_data[m*3 + 1]),
                        X = std::abs((real_t) x - cms_data[m*3 + 2]);

                    real_t
                        a = abc_data[m*3 + 0],
                        b = abc_data[m*3 + 1],
                        c = abc_data[m*3 + 2];

                    //double value = std::pow(Z/a, 2) + std::pow(Y/b, 2) + std::pow(X/c, 2);
                    double value = ((Z*Z)/(a*a)) + ((Y*Y)/(b*b)) + ((X*X)/(c*c));

                    if (value >= 1) {
                        #pragma omp atomic
                        errors_data[m]++;
                    }
                }
            }
        }
    }

    template <typename T>
    void sample_plane(const input_ndarray<T> &voxels,
            const real_t voxel_size,
            const std::array<real_t, 3> cm,
            const std::array<real_t, 3> u_axis,
            const std::array<real_t, 3> v_axis,
            const std::array<real_t, 4> bbox,
            output_ndarray<real_t> plane_samples,
            const int verbose) {
        const auto& [umin,umax,vmin,vmax] = bbox; // In micrometers
        UNPACK_NUMPY(voxels);
        ssize_t
            nu = plane_samples.shape[0],
            nv = plane_samples.shape[1];
        real_t
            du = (umax - umin) / real_t(nu),
            dv = (vmax - vmin) / real_t(nv);

        real_t *dat = plane_samples.data;

        #pragma acc data copyin(voxels, voxels.data[:voxels_Nz*voxels_Ny*voxels_Nx], voxels_Nz, voxels_Ny, voxels_Nx, cm[:3], u_axis[:3], v_axis[:3]) create(dat[:nu*nv]) copyout(dat[:nu*nv])
        {
        PRAGMA(PARALLEL_TERM() collapse(2))
        for (ssize_t ui = 0; ui < nu; ui++) {
            for (ssize_t vj = 0; vj < nv; vj++) {
                const real_t
                    u = umin + real_t(ui)*du,
                    v = vmin + real_t(vj)*dv;

                // X,Y,Z in micrometers;  x,y,z in voxel index space
                const real_t
                    Z = cm[0] + u*u_axis[0] + v*v_axis[0],
                    Y = cm[1] + u*u_axis[1] + v*v_axis[1],
                    X = cm[2] + u*u_axis[2] + v*v_axis[2];

                const real_t
                    z = Z / voxel_size,
                    x = X / voxel_size,
                    y = Y / voxel_size;

                if (verbose >= 2) {
                    printf("u,v = %g,%g -> %.1f,%.1f,%.1f -> %d, %d, %d\n", u, v, X, Y, Z, int(round(x)), int(round(y)), int(round(z)));
                }

                T value = 0;
                std::array<float, 6> local_bbox = {0.5f, float(voxels_Nz)-0.5f, 0.5f, float(voxels_Ny)-0.5f, 0.5f, float(voxels_Nx)-0.5f};
                if (in_bbox({{z,y,x}}, local_bbox)) {
                    value = (T) std::round(resample2x2x2<T>(voxels.data, {voxels_Nz, voxels_Ny, voxels_Nx}, {z, y, x}));
                }
                else if (verbose >= 2) {
                    fprintf(stderr, "Sampling outside image: x,y,z = %.1f,%.1f,%.1f, Nx,Ny,Nz = %ld,%ld,%ld\n", x, y, z, voxels_Nx, voxels_Ny, voxels_Nz);
                }

                dat[ui*nv + vj] = value;
            }
        }
        }
    }

    // NB: xyz are in indices, not micrometers
    void zero_outside_bbox(const std::array<real_t,9> &principal_axes,
            const std::array<real_t,6> &parameter_ranges,
            const std::array<real_t, 3> &cm,
            output_ndarray<mask_type> mask) {

        UNPACK_NUMPY(mask)

        #pragma acc data copyin(principal_axes, parameter_ranges, cm)
        {
            BLOCK_BEGIN(mask, ) {

                real_t xs[3] = {
                    real_t(x) - cm[0],
                    real_t(y) - cm[1],
                    real_t(z) - cm[2]};
                real_t params[3] = { 0, 0, 0 };

                for (int uvw = 0; uvw < 3; uvw++) {
                    for (int xyz = 0; xyz < 3; xyz++) {
                        params[uvw] += xs[xyz] * principal_axes[uvw*3 + xyz]; // u = dot(xs,u_axis), v = dot(xs,v_axis), w = dot(xs,w_axis)
                    }
                }

                bool p = false;

                for (int uvw = 0; uvw < 3; uvw++) {
                    real_t
                        param_min = parameter_ranges[uvw*2],
                        param_max = parameter_ranges[uvw*2 + 1];
                    p |= (params[uvw] < param_min) | (params[uvw] > param_max);
                }

                if (p) {
                    mask_buffer[flat_index] = 0;
                }

            BLOCK_END() }
        }
    }

}
