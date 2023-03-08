// TODO: Coordinates are named X,Y,Z in c++, but Z,Y,X in python. Homogenize to X,Y,Z!
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <math.h>

#include "boilerplate.hh"
#include "geometry.hh"

using namespace std;
namespace cpu_seq {

array<real_t, 3> center_of_mass(const input_ndarray<mask_type> &mask) {
    UNPACK_NUMPY(mask);

    print_timestamp("center_of_mass start");

    uint64_t total_mass = 0, cmz = 0, cmy = 0, cmx = 0;

    BLOCK_BEGIN(mask, reduction(+:total_mass,cmz,cmy,cmx)); {
    // TODO James approves; now RUN!

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

    print_timestamp("center_of_mass end");

    return array<real_t, 3>{ rcmz, rcmy, rcmx };
}

void fill_implant_mask(const input_ndarray<mask_type> mask,
               float voxel_size,
               const array<float,6> &bbox,
               float r_fraction,
               const matrix4x4 &Muvw,
               output_ndarray<mask_type> solid_implant_mask,
               output_ndarray<float> rsqr_maxs,
               output_ndarray<float> profile) {
    UNPACK_NUMPY(mask)

    real_t theta_min = real_t(M_PI), theta_max = real_t(-M_PI);
    ssize_t n_segments = rsqr_maxs.shape[0];
    const auto [U_min,U_max,V_min,V_max,W_min,W_max] = bbox;
    float     *rsqr_maxs_d     = rsqr_maxs.data;
    float     *profile_d       = profile.data;


    #pragma omp parallel for collapse(3) reduction(max:rsqr_maxs_d[:n_segments], theta_max) reduction(min:theta_min)
    for (int64_t z = 0; z < mask_Nz; z++) {
        for (int64_t y = 0; y < mask_Ny; y++) {
            for (int64_t x = 0; x < mask_Nx; x++) {
        mask_type mask_value = mask.data[z*mask_Ny*mask_Nx + y*mask_Nx + x];
        std::array<real_t, 4> Xs = {
            real_t(x) * voxel_size,
            real_t(y) * voxel_size,
            real_t(z) * voxel_size,
            1 };

        if (mask_value) {
            auto [U,V,W,c] = hom_transform(Xs, Muvw);

                    real_t r_sqr = V*V + W*W;
                    real_t theta = atan2(V, W);

                    int U_i = int(floor((U - U_min) * real_t(n_segments-1) / (U_max - U_min)));

        //    if (U_i >= 0 && U_i < n_segments) {
                    if ( in_bbox(U, V, W, bbox) ) {
                rsqr_maxs_d[U_i] = max(rsqr_maxs_d[U_i], float(r_sqr));
                theta_min = min(theta_min, theta);
                theta_max = max(theta_max, theta);
            //      W_min     = min(W_min,     W);
            } else {
                // Otherwise we've calculated it wrong!
                //  fprintf(stderr,"U-coordinate out of bounds: U_i = %ld, U = %g, U_min = %g, U_max = %g\n",U_i,U,U_min,U_max);
            }
        }
            }
        }
    }

    real_t theta_center = (theta_max + theta_min) / 2;

    #pragma omp parallel for collapse(3) reduction(+:profile_d[:n_segments])
    for (int64_t z = 0; z < mask_Nz; z++) {
        for (int64_t y = 0; y < mask_Ny; y++) {
            for (int64_t x = 0; x < mask_Nx; x++) {
        std::array<real_t, 4> Xs = {
            real_t(x) * voxel_size,
            real_t(y) * voxel_size,
            real_t(z) * voxel_size,
            1 };
        int64_t flat_index = z*mask_Ny*mask_Nx + y*mask_Nx + x;
        mask_type mask_value = mask.data[flat_index];

        // Second pass does the actual work
                auto [U,V,W,c] = hom_transform(Xs, Muvw);
                float r_sqr = V*V + W*W;
                float theta = atan2(V, W);
                int U_i = int(floor((U - U_min) * real_t(n_segments-1) / (U_max - U_min)));

        bool solid_mask_value = false;
        if (U_i >= 0 && U_i < n_segments && W >= W_min) { // TODO: Full bounding box check?
                    solid_mask_value = mask_value | (r_sqr <= r_fraction * rsqr_maxs_d[U_i]);

            if (theta >= theta_min && theta <= theta_center && r_sqr <= rsqr_maxs_d[U_i]) {
                profile_d[U_i] += solid_mask_value;
            }
        }

        solid_implant_mask.data[flat_index] = solid_mask_value;
            }
        }
    }
}

array<real_t,9> inertia_matrix(const input_ndarray<mask_type> &mask, const array<real_t,3> &cm) {
    UNPACK_NUMPY(mask);

    real_t
        Ixx = 0, Ixy = 0, Ixz = 0,
                 Iyy = 0, Iyz = 0,
                          Izz = 0;

    print_timestamp("inertia_matrix_serial start");

    BLOCK_BEGIN(mask, reduction(+:Ixx, Iyy, Izz) reduction(-:Ixy,Ixz,Iyz)) {

        mask_type m = mask_buffer[flat_index];

        // m guards this, and then branches are removed
        //if (m != 0)
        real_t
            X = real_t(x) - cm[0],
            Y = real_t(y) - cm[1],
            Z = real_t(z) - cm[2];

        Ixx += m * (Y*Y + Z*Z);
        Iyy += m * (X*X + Z*Z);
        Izz += m * (X*X + Y*Y);
        Ixy -= m * X*Y;
        Ixz -= m * X*Z;
        Iyz -= m * Y*Z;

    } BLOCK_END();

    print_timestamp("inertia_matrix_serial end");

    return array<real_t,9> {
        Ixx, Ixy, Ixz,
        Ixy, Iyy, Iyz,
        Ixz, Iyz, Izz
    };
}

template <typename T>
float resample2x2x2(const T             *voxels,
                    const array<ssize_t, 3> &shape,
                    const array<float, 3>   &X) {
    auto  [Nz,Ny,Nx] = shape;

    if (!in_bbox(X[0], X[1], X[2], {0.5f, float(Nx)-0.5f, 0.5f, float(Ny)-0.5f, 0.5f, float(Nz)-0.5f})) {
        uint64_t voxel_index = uint64_t(floor(X[0]))*Ny*Nz + uint64_t(floor(X[1]))*Ny + uint64_t(floor(X[2]));
        return voxels[voxel_index];
    }

    float   Xfrac[2][3]; // {Xminus[3], Xplus[3]}
    int64_t Xint[2][3];  // {Iminus[3], Iplus[3]}
    float   value = 0;

    for (int i = 0; i < 3; i++) {
        float Iminus, Iplus;
        Xfrac[0][i] = 1-modf(X[i]-0.5f, &Iminus); // 1-{X[i]-1/2}, floor(X[i]-1/2)
        Xfrac[1][i] =   modf(X[i]+0.5f, &Iplus);  // {X[i]+1/2}, floor(X[i]+1/2)

        Xint[0][i] = (int64_t) Iminus;
        Xint[1][i] = (int64_t) Iplus;
    }

    for (int ijk = 0; ijk <= 7; ijk++) {
        float  weight = 1;
        int64_t IJK[3] = {0,0,0};

        for (int axis = 0; axis < 3; axis++) { // x-1/2 or x+1/2
            int pm    = (ijk >> axis) & 1;
            IJK[axis] = Xint[pm][axis];
            weight   *= Xfrac[pm][axis];
        }

        auto [I,J,K] = IJK;
        // if (I<0 || J<0 || K<0) {
        //   printf("(I,J,K) = (%ld,%ld,%ld)\n",I,J,K);
        //   abort();
        // }
        // if (I>=int(Nx) || J>=int(Ny) || K>=int(Nz)) {
        //   printf("(I,J,K) = (%ld,%ld,%ld), (Nx,Ny,Nz) = (%ld,%ld,%ld)\n",I,J,K,Nx,Ny,Nz);
        //   abort();
        // }
        uint64_t voxel_index = I*Ny*Nz+J*Ny+K;
        //assert(I>=0 && J>=0 && K>=0);
        //assert(I<Nx && J<Ny && K<Nz);
        float voxel = (float) voxels[voxel_index];
        value += voxel*weight;
    }

    return value;
}

template <typename T>
void sample_plane(const input_ndarray<T> &voxels,
                  const real_t voxel_size, // In micrometers
                  const array<real_t, 3> cm,
                  const array<real_t, 3> u_axis,
                  const array<real_t, 3> v_axis,
                  const array<real_t, 4> bbox,    // [umin,umax,vmin,vmax] in micrometers
                  output_ndarray<real_t> plane_samples) {
    const auto& [umin,umax,vmin,vmax] = bbox; // In micrometers
    UNPACK_NUMPY(voxels);
    ssize_t
        nu = plane_samples.shape[0],
        nv = plane_samples.shape[1];
    real_t
        du = (umax - umin) / real_t(nu),
        dv = (vmax - vmin) / real_t(nv);

    real_t *dat = plane_samples.data;

    #pragma acc data copyin(voxels, voxels.data[:voxels_Nz*voxels_Ny*voxels_Nx], voxels_Nz, voxels_Ny, voxels_Nx) create(dat[:nu*nv]) copyout(dat[:nu*nv])
    {
    PRAGMA(PARALLEL_TERM collapse(2))
    for (ssize_t ui = 0; ui < nu; ui++) {
        for (ssize_t vj = 0; vj < nv; vj++) {
            const real_t
                u = umin + real_t(ui)*du,
                v = vmin + real_t(vj)*dv;

            // X,Y,Z in micrometers;  x,y,z in voxel index space
            const real_t
                X = cm[0] + u*u_axis[0] + v*v_axis[0],
                Y = cm[1] + u*u_axis[1] + v*v_axis[1],
                Z = cm[2] + u*u_axis[2] + v*v_axis[2];

            const real_t
                x = X / voxel_size,
                y = Y / voxel_size,
                z = Z / voxel_size;

            //      printf("u,v = %g,%g -> %.1f,%.1f,%.1f -> %d, %d, %d\n",u,v,X,Y,Z,int(round(x)),int(round(y)),int(round(z)));

            T value = 0;
            std::array<float, 6> local_bbox = {0.5f, float(voxels_Nx)-0.5f, 0.5f, float(voxels_Ny)-0.5f, 0.5f, float(voxels_Nz)-0.5f};
            if (in_bbox(x,y,z, local_bbox))
                value = (T) round(resample2x2x2<T>(voxels.data, {voxels_Nx, voxels_Ny, voxels_Nz}, {x, y, z}));
            // else
            //     fprintf(stderr,"Sampling outside image: x,y,z = %.1f,%.1f,%.1f, Nx,Ny,Nz = %ld,%ld,%ld\n",x,y,z,Nx,Ny,Nz);

            dat[ui*nv + vj] = value;
        }
    }
    }
}

void integrate_axes(const input_ndarray<mask_type> &mask,
		    const array<real_t,3> &x0,
		    const array<real_t,3> &v_axis,
		    const array<real_t,3> &w_axis,
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
                v = dot(xs, v_axis),
                w = dot(xs, w_axis);
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

// NB: xyz are in indices, not micrometers
void zero_outside_bbox(const array<real_t,9> &principal_axes,
               const array<real_t,6> &parameter_ranges,
               const array<real_t,3> &cm,
               output_ndarray<mask_type> voxels) {

    UNPACK_NUMPY(voxels)

    #pragma acc data copyin(principal_axes, parameter_ranges, cm)
    {
    BLOCK_BEGIN(voxels, ) {

        real_t xs[3] = {
            real_t(x) - cm[0],
            real_t(y) - cm[1],
            real_t(z) - cm[2]};
        real_t params[3] = { 0, 0, 0 };

        for (int uvw = 0; uvw < 3; uvw++)
            for (int xyz = 0; xyz < 3; xyz++)
                params[uvw] += xs[xyz] * principal_axes[uvw*3 + xyz]; // u = dot(xs,u_axis), v = dot(xs,v_axis), w = dot(xs,w_axis)

        bool p = false;

        for (int uvw = 0; uvw < 3; uvw++) {
            real_t
                param_min = parameter_ranges[uvw*2],
                param_max = parameter_ranges[uvw*2 + 1];
            p |= (params[uvw] < param_min) | (params[uvw] > param_max);
        }

        if (p)
            voxels_buffer[flat_index] = 0;

    BLOCK_END() }
    }
}

}

/*
void compute_front_mask(const input_ndarray<mask_type> solid_implant,
        const float voxel_size,
        const matrix4x4 &Muvw,
        std::array<float,6> bbox,
        output_ndarray<mask_type> front_mask) {
    const auto [U_min,U_max,V_min,V_max,W_min,W_max] = bbox;

    loop_mask_start(solid_implant, front_mask, () );

    if (!mask_value) {
        auto [U,V,W,c] = hom_transform(Xs,Muvw);
        maskout_buffer[k] = W>W_min;
    } else
        maskout_buffer[k] = 0;

    loop_mask_end(solid_implant)
}

void cylinder_projection(const input_ndarray<float>  edt,  // Euclidean Distance Transform in um, should be low-resolution (will be interpolated)
             const input_ndarray<uint8_t> C,  // Material classification images (probability per voxel, 0..1 -> 0..255)
             float voxel_size,           // Voxel size for Cs
             float d_min, float d_max,       // Distance shell to map to cylinder
             float theta_min, float theta_max, // Angle range (wrt cylinder center)
             std::array<float,6> bbox,
             const matrix4x4 &Muvw,           // Transform from zyx (in um) to U'V'W' cylinder FoR (in um)
             output_ndarray<float>    image,  // Probability-weighted volume of (class,theta,U)-voxels
             output_ndarray<int64_t>  count   // Number of (class,theta,U)-voxels
             ){
    ssize_t n_theta = image.shape[0], n_U = image.shape[1];

    const auto& [U_min,U_max,V_min,V_max,W_min,W_max] = bbox;

    ssize_t ex = edt.shape[0], ey = edt.shape[1], ez = edt.shape[2];
    ssize_t Cx = C.shape[0],   Cy = C.shape[1],   Cz = C.shape[2];

    real_t edx = ex/real_t(Cx), edy = ey/real_t(Cy), edz = ex/real_t(Cz);

    ssize_t edt_length       = ex*ey*ez;
    ssize_t C_length         = Cx*Cy*Cz;

    printf("Segmenting from %g to %g micrometers distance of implant.\n",d_min,d_max);

    printf("Bounding box is [U_min,U_max,V_min,V_max,W_min,W_max] = [[%g,%g],[%g,%g],[%g,%g]]\n",
        U_min,U_max,V_min,V_max,W_min,W_max);
    printf("EDT field is (%ld,%ld,%ld)\n",ex,ey,ez);

    real_t th_min = 1234, th_max = -1234;
    ssize_t n_shell = 0;
    ssize_t n_shell_bbox = 0;

    ssize_t block_height = 64;

    //TODO: new acc/openmp macro in parallel.hh
    {
        float   *image_d = image.data;
        int64_t *count_d = count.data;

        for (ssize_t block_start = 0, edt_block_start = 0; block_start < C_length; block_start += block_height*Cy*Cz, edt_block_start += block_height*ey*ez) {
            const uint8_t *C_buffer = C.data + block_start;
            const float  *edt_block = edt.data + max(block_start-ey*ez,0L);

            ssize_t  this_block_length = min(block_height*Cy*Cz,C_length-block_start);
            ssize_t  this_edt_length   = min((block_height+2)*ey*ez,edt_length-block_start);

            //#pragma acc parallel loop copy(C_buffer[:this_block_length], image_d[:n_theta*n_U], count_d[:n_theta*n_U], bbox[:6], Muvw[:16], edt_block[:this_edt_length]) reduction(+:n_shell,n_shell_bbox)
            #pragma omp parallel for reduction(+:n_shell,n_shell_bbox)
            for (int64_t k = 0; k < this_block_length; k++) {
                const int64_t flat_idx = block_start + k;
                const int64_t X = (flat_idx  / (Cy*Cz)), Y = (flat_idx / Cz) % Cy, Z = flat_idx  % Cz; // Integer indices: Cs[c,X,Y,Z]
                // Index into local block
                const int64_t Xl = (k  / (Cy*Cz)), Yl = (k / Cz) % Cy, Zl = k  % Cz;
                // Index into local edt block. Note EDT has 1-slice padding top+bottom
                const float  x = (Xl+1)*edx, y = Yl*edy, z = Zl*edy;

                if (x > block_height) {
                    printf("Block number k=%ld.\nX,Y,Z=%ld,%ld,%ld\nXl,Yl,Zl=%ld,%ld,%ld\nx,y,z=%.2f, %.2f, %.2f\n",k,X,Y,Z,Xl,Yl,Zl,x,y,z);
                    abort();
                }

                // ****** MEAT OF THE IMPLEMENTATION IS HERE ******
                real_t distance = resample2x2x2<float>(edt_block, {this_edt_length/(ey*ez),ey,ez}, {x,y,z});

                if (distance > d_min && distance <= d_max) { // TODO: and W>w_min
                    array<real_t,4> Xs = {X*voxel_size, Y*voxel_size, Z*voxel_size, 1};
                    auto [U,V,W,c] = hom_transform(Xs,Muvw);
                    n_shell ++;

                    //        printf("distance = %.1f, U,V,W = %.2f,%.2f,%.2f\n",distance,U,V,W);
                    if (in_bbox(U,V,W,bbox)) {
                        real_t theta    = atan2(V,W);

                        if (theta >= theta_min && theta <= theta_max) {
                            n_shell_bbox++;

                            ssize_t theta_i = floor( (theta-theta_min) * (n_theta-1)/(theta_max-theta_min) );
                            ssize_t U_i     = floor( (U    -    U_min) * (n_U    -1)/(    U_max-    U_min) );

                            real_t p = C_buffer[k]/255.;

                            assert(theta >= theta_min);
                            assert(theta <= theta_max);
                            assert(U >= U_min);
                            assert(U <= U_max);
                            assert(theta_i >= 0);
                            assert(theta_i < n_theta);
                            assert(U_i >= 0);
                            assert(U_i < n_U);

                            if (p > 0) {
                                th_min = min(theta,th_min);
                                th_max = max(theta,th_max);

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
    printf("n_shell = %ld, n_shell_bbox = %ld\n",n_shell,n_shell_bbox);
    printf("theta_min, theta_max = %.2f,%.2f\n",theta_min,theta_max);
    printf("th_min,       th_max = %.2f,%.2f\n",th_min,th_max);
}

*/