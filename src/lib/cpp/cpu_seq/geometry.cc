// TODO: Coordinates are named X,Y,Z in c++, but Z,Y,X in python. Homogenize to X,Y,Z!
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <math.h>
using namespace std;

#include "geometry.hh"
#include "boilerplate.hh"

array<real_t, 3> center_of_mass(const input_ndarray<mask_type> voxels) {
    unpack_numpy(voxels);

    print_timestamp("center_of_mass start");

    uint64_t cmz = 0, cmy = 0, cmx = 0;
    uint64_t total_mass = 0;

    for_3d_begin(voxels);

    mask_type m = voxels.data[flat_index];

    total_mass += m;
    cmx += m * x;
    cmy += m * y;
    cmz += m * z;

    for_3d_end();

    real_t
        rcmx = cmx / ((real_t) total_mass),
        rcmy = cmy / ((real_t) total_mass),
        rcmz = cmz / ((real_t) total_mass);

    print_timestamp("center_of_mass end");

    return array<real_t, 3>{ rcmz, rcmy, rcmx };
}

array<real_t,9> inertia_matrix(const input_ndarray<mask_type> &voxels, const array<real_t,3> &cm) {
    real_t
        Ixx = 0, Ixy = 0, Ixz = 0,
                 Iyy = 0, Iyz = 0,
                          Izz = 0;

    ssize_t Nx = voxels.shape[0], Ny = voxels.shape[1], Nz = voxels.shape[2];

    print_timestamp("inertia_matrix_serial start");

    int64_t k = 0;
    for (int64_t X = 0; X < Nx; X++) {
        for (int64_t Y = 0; Y < Ny; Y++) {
            for (int64_t Z = 0; Z < Nz; Z++) {
                mask_type m = voxels.data[k];
                k++;

                // m guards this, and then branches are removed
                //if (m != 0)
                real_t
                    x = X - cm[0],
                    y = Y - cm[1],
                    z = Z - cm[2];

                Ixx += m * (y*y + z*z);
                Iyy += m * (x*x + z*z);
                Izz += m * (x*x + y*y);
                Ixy -= m * x*y;
                Ixz -= m * x*z;
                Iyz -= m * y*z;
            }
        }
    }

    print_timestamp("inertia_matrix_serial end");

    return array<real_t,9> {
        Ixx, Ixy, Ixz,
        Ixy, Iyy, Iyz,
        Ixz, Iyz, Izz
    };
}

/* TODO only called in test.py. Postponed for now.
void integrate_axes(const input_ndarray<mask_type> &voxels,
            const array<real_t,3> &x0,
            const array<real_t,3> &v_axis,
            const array<real_t,3> &w_axis,
            const real_t v_min, const real_t w_min,
            output_ndarray<real_t> output) {
    ssize_t Nx = voxels.shape[0], Ny = voxels.shape[1], Nz = voxels.shape[2];
    ssize_t Nv = output.shape[0], Nw = output.shape[1];
    int64_t image_length = Nx*Ny*Nz;
    real_t *output_data = output.data;

    // TODO: Check v_axis & w_axis projections to certify bounds and get rid of runtime check
    int64_t k = 0:
    for (int64_t X = 0; X < Nx; X++) {
        for (int64_t Y = 0; Y < Ny; Y++) {
            for (int64_t Z = 0; Z < Nz; Z++) {
                if (buffer[k] != 0) {
                    real_t xs[3] = {
                        (flat_idx  / (Ny*Nz))  - x0[0],   // x
                        ((flat_idx / Nz) % Ny) - x0[1],   // y
                        (flat_idx  % Nz)       - x0[2] }; // z

                    mask_type voxel = buffer[k];
                    real_t v = dot(xs, v_axis), w = dot(xs,w_axis);
                    int64_t i_v = round(v-v_min), j_w = round(w-w_min);

                    if (i_v >= 0 && j_w >= 0 && i_v < Nv && j_w < Nw) {
                        output_data[i_v*Nw + j_w] += voxel;
                    }
                }
                k++;
            }
        }
    }
}
*/

bool in_bbox(float U, float V, float W, const std::array<float,6> bbox) {
    const auto& [U_min,U_max,V_min,V_max,W_min,W_max] = bbox;

    bool inside = U>=U_min && U<=U_max && V>=V_min && V<=V_max && W>=W_min && W<=W_max;

    // printf("in_bbox: (%.1f,%.1f,%.1f) \in ([%.1f,%.1f],[%.1f,%.1f],[%.1f,%.1f]) == %d\n",
    //      U,V,W,U_min,U_max,V_min,V_max,U_min,U_max,inside);
    return inside;
}

template<typename field_type> float resample2x2x2(const field_type *voxels,
                                                  const array<ssize_t,3> &shape,
                                                  const array<float,3>   &X) {
    auto  [Nx,Ny,Nz] = shape;    // Eller omvendt?
    if (!in_bbox(X[0],X[1],X[2], {0.5,Nx-1.5, 0.5,Ny-1.5, 0.5,Nz-1.5})) {
        uint64_t voxel_index = floor(X[0])*Ny*Nz+floor(X[1])*Ny+floor(X[2]);
        return voxels[voxel_index];
    }
    float   Xfrac[2][3]; // {Xminus[3], Xplus[3]}
    int64_t Xint[2][3];     // {Iminus[3], Iplus[3]}
    float   value = 0;

    for (int i = 0; i < 3; i++) {
        double Iminus, Iplus;
        Xfrac[0][i] = 1-modf(X[i]-0.5, &Iminus); // 1-{X[i]-1/2}, floor(X[i]-1/2)
        Xfrac[1][i] =   modf(X[i]+0.5, &Iplus);  // {X[i]+1/2}, floor(X[i]+1/2)

        Xint[0][i] = Iminus;
        Xint[1][i] = Iplus;
    }


    for (int ijk = 0; ijk <= 7; ijk++) {
        float  weight = 1;
        int64_t IJK[3] = {0,0,0};

        for (int axis = 0; axis < 3; axis++) { // x-1/2 or x+1/2
            int pm = (ijk>>axis) & 1;
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
        assert(I>=0 && J>=0 && K>=0);
        assert(I<Nx && J<Ny && K<Nz);
        field_type voxel = voxels[voxel_index];
        value += voxel*weight;
    }
    return value;
}

template <typename voxel_type> void sample_plane(const input_ndarray<voxel_type> &voxels,
                         const real_t voxel_size, // In micrometers
                         const array<real_t,3> cm,
                         const array<real_t,3> u_axis,
                         const array<real_t,3> v_axis,
                         const array<real_t,4>  bbox,    // [umin,umax,vmin,vmax] in micrometers
                         output_ndarray<real_t> plane_samples) {
    const auto& [umin,umax,vmin,vmax] = bbox; // In micrometers
    ssize_t Nx = voxels.shape[0], Ny = voxels.shape[1], Nz = voxels.shape[2];
    ssize_t nu = plane_samples.shape[0], nv = plane_samples.shape[1];
    real_t  du = (umax-umin)/nu, dv = (vmax-vmin)/nv;

    #pragma omp parallel for collapse(2)
    for (ssize_t ui=0;ui<nu;ui++) {
        for (ssize_t vj=0;vj<nv;vj++) {
            const real_t u = umin + ui*du, v = vmin + vj*dv;

            // X,Y,Z in micrometers;  x,y,z in voxel index space
            const real_t
                X = cm[0] + u*u_axis[0] + v*v_axis[0],
                Y = cm[1] + u*u_axis[1] + v*v_axis[1],
                Z = cm[2] + u*u_axis[2] + v*v_axis[2];

            const real_t x = X/voxel_size, y = Y/voxel_size, z = Z/voxel_size;

            //      printf("u,v = %g,%g -> %.1f,%.1f,%.1f -> %d, %d, %d\n",u,v,X,Y,Z,int(round(x)),int(round(y)),int(round(z)));

            voxel_type value = 0;
            if (in_bbox(x,y,z,{0.5,Nx-0.5, 0.5,Ny-0.5, 0.5,Nz-0.5}))
                value = resample2x2x2<voxel_type>(voxels.data,{Nx,Ny,Nz},{x,y,z});
            // else
            //     fprintf(stderr,"Sampling outside image: x,y,z = %.1f,%.1f,%.1f, Nx,Ny,Nz = %ld,%ld,%ld\n",x,y,z,Nx,Ny,Nz);

            plane_samples.data[ui*nv + vj] = value;
        }
    }
}

/* TODO only called in test.py. Postpone for now.
// NB: xyz are in indices, not micrometers
void zero_outside_bbox(const array<real_t,9> &principal_axes,
               const array<real_t,6> &parameter_ranges,
               const array<real_t,3> &cm,
               output_ndarray<mask_type> voxels) {
    size_t  Nx = voxels.shape[0], Ny = voxels.shape[1], Nz = voxels.shape[2];
    int64_t image_length = Nx*Ny*Nz;

    printf("(Nx,Ny,Nz) = (%ld,%ld,%ld), image_length = %ld",Nx,Ny,Nz, image_length);

    for (int64_t block_start = 0; block_start < image_length; block_start += acc_block_size) {
        mask_type *buffer = voxels.data + block_start;
        ssize_t this_block_length = min(acc_block_size, image_length-block_start);

        //parallel_loop((buffer[:this_block_length]))
        for (int64_t k = 0; k < this_block_length; k++) {
            int64_t flat_idx = block_start + k;
            int64_t x = flat_idx  / (Ny*Nz);
            int64_t y = (flat_idx / Nz) % Ny;
            int64_t z = flat_idx  % Nz;
            // Boilerplate until here. TODO: macroize or lambda out!

            real_t xs[3] = {x-cm[0], y-cm[1], z-cm[2]};

            real_t params[3] = {0,0,0};

            for (int uvw = 0; uvw < 3; uvw++)
                for (int xyz = 0; xyz < 3; xyz++)
                    params[uvw] += xs[xyz] * principal_axes[uvw*3+xyz]; // u = dot(xs,u_axis), v = dot(xs,v_axis), w = dot(xs,w_axis)

            bool p = false;

            for (int uvw = 0; uvw < 3; uvw++) {
                real_t param_min = parameter_ranges[uvw*2], param_max = parameter_ranges[uvw*2+1];
                p |= (params[uvw] < param_min) | (params[uvw] > param_max);
            }

            if (p) buffer[k] = 0;

        }
    }
}
*/

inline vector4 hom_transform(const vector4 &x, const matrix4x4 &M) {
    vector4 c{{0,0,0,0}};

    for (int i = 0; i < 4; i++) {
        real_t sum = 0;
        #pragma simd parallel for reduction(+:sum)
        for (int j=0;j<4;j++)
            sum += M[i*4+j]*x[j];
        c[i] = sum;
    }
    return c;
}

#define loop_mask_start(mask_in,mask_out,COPY) {                                                                                \
    ssize_t Mx = mask_in.shape[0], My = mask_in.shape[1], Mz = mask_in.shape[2];                                                \
    ssize_t mask_length = Mx*My*Mz;                                                                                             \
                                                                                                                                \
    for (ssize_t block_start = 0; block_start < mask_length; block_start += acc_block_size) {                                   \
        const mask_type *maskin_buffer  = mask_in.data + block_start;                                                           \
            mask_type *maskout_buffer = mask_out.data + block_start;                                                            \
        ssize_t this_block_length = min(acc_block_size, mask_length-block_start);                                               \
                                                                                                                                \
        _Pragma(STR(acc parallel loop copy(maskin_buffer[:this_block_length], maskout_buffer[:this_block_length]) copy COPY))   \
        for (int64_t k = 0; k < this_block_length; k++) {                                                                       \
            int64_t flat_idx = block_start + k;                                                                                 \
            int64_t X = (flat_idx  / (My*Mz)), Y = (flat_idx / Mz) % My, Z = flat_idx  % Mz;                                    \
            std::array<real_t,4> Xs = { X*voxel_size, Y*voxel_size, Z*voxel_size, 1 };                                          \
            bool mask_value = maskin_buffer[k];

#define loop_mask_end(mask) }}}

/*
void fill_implant_mask(const input_ndarray<mask_type> implant_mask,
               float voxel_size,
               const array<float,6> &bbox,
               float r_fraction,
               const matrix4x4 &Muvw,
               output_ndarray<mask_type> solid_implant_mask,
               output_ndarray<float> rsqr_maxs,
               output_ndarray<float> profile) {
    real_t theta_min = M_PI, theta_max = -M_PI;
    ssize_t n_segments = rsqr_maxs.shape[0];
    const auto [U_min,U_max,V_min,V_max,W_min,W_max] = bbox;

    printf("implant_mask.shape = %ld,%ld,%ld\n",implant_mask.shape[0],implant_mask.shape[1],implant_mask.shape[2]);
    printf("solid_implant_mask.shape = %ld,%ld,%ld\n",solid_implant_mask.shape[0],solid_implant_mask.shape[1],solid_implant_mask.shape[2]);

    fprintf(stderr,"voxel_size = %g, U_min = %g, U_max = %g, r_frac = %g, n_segments = %ld\n",
        voxel_size, U_min, U_max, r_fraction, n_segments);

    float     *rsqr_maxs_d     = rsqr_maxs.data;
    float     *profile_d       = profile.data;

    // First pass computes some bounds -- possibly separate out to avoid repeating
    //loop_mask_start(implant_mask, solid_implant_mask, (maskin_buffer[:this_block_length], rsqr_maxs_d[:n_segments], Muvw[:16], bbox[:6]) );
    if (mask_value) {
        auto [U,V,W,c] = hom_transform(Xs,Muvw);

        real_t r_sqr = V*V+W*W;
        real_t theta = atan2(V,W);

        int U_i = floor((U-U_min)*(n_segments-1)/(U_max-U_min));

        //    if (U_i >= 0 && U_i < n_segments) {
        if ( in_bbox(U,V,W,bbox) ) {
            rsqr_maxs_d[U_i] = max(rsqr_maxs_d[U_i], float(r_sqr));
            theta_min = min(theta_min, theta);
            theta_max = max(theta_max, theta);
            //      W_min     = min(W_min,     W);
        } else {
            // Otherwise we've calculated it wrong!
            //  fprintf(stderr,"U-coordinate out of bounds: U_i = %ld, U = %g, U_min = %g, U_max = %g\n",U_i,U,U_min,U_max);
        }
    }
    //loop_mask_end(implant_mask);

    double theta_center = (theta_max+theta_min)/2;

    fprintf(stderr,"theta_min, theta_center, theta_max = %g,%g,%g\n", theta_min, theta_center, theta_max);

    // Second pass does the actual work
    //loop_mask_start(implant_mask, solid_implant_mask,
            (rsqr_maxs_d[:n_segments], profile_d[:n_segments]) );
    auto [U,V,W,c] = hom_transform(Xs,Muvw);
    float r_sqr = V*V+W*W;
    float theta = atan2(V,W);
    int U_i = floor((U-U_min)*(n_segments-1)/(U_max-U_min));

    bool solid_mask_value = false;
    if (U_i >= 0 && U_i < n_segments && W >= W_min) { // TODO: Full bounding box check?
        solid_mask_value = mask_value | (r_sqr <= r_fraction*rsqr_maxs_d[U_i]);

        if (theta >= theta_min && theta <= theta_center && r_sqr <= rsqr_maxs_d[U_i]) {
            //atomic_statement()
            profile_d[U_i] += solid_mask_value;
        }
    }
    maskout_buffer[k] = solid_mask_value;

    //loop_mask_end(implant_mask);
}

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
*/

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

                //****** MEAT OF THE IMPLEMENTATION IS HERE ******
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
