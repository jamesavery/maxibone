#ifndef geometry_h
#define geometry_h

using namespace std;

#include "datatypes.hh"
#include <chrono>
#include <string>

#define dot(a,b) (a[0]*b[0] + a[1]*b[1] + a[2]*b[2])

inline void print_timestamp(string message) {
    auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    tm local_tm = *localtime(&now);
    fprintf(stderr,"%s at %02d:%02d:%02d\n", message.c_str(), local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
}

inline vector4 hom_transform(const vector4 &x, const matrix4x4 &M) {
    vector4 c{{ 0, 0, 0, 0 }};

    for (int i = 0; i < 4; i++) {
        real_t sum = 0;
        for (int j = 0; j < 4; j++)
            sum += M[i*4 + j] * x[j];
        c[i] = sum;
    }
    return c;
}

inline bool in_bbox(float U, float V, float W, const std::array<float, 6> &bbox) {
    const auto& [U_min, U_max, V_min, V_max, W_min, W_max] = bbox;

    bool inside =
        U >= U_min &&
        U <= U_max &&
        V >= V_min &&
        V <= V_max &&
        W >= W_min &&
        W <= W_max;

    // printf("in_bbox: (%.1f,%.1f,%.1f) \in ([%.1f,%.1f],[%.1f,%.1f],[%.1f,%.1f]) == %d\n",
    //      U,V,W,U_min,U_max,V_min,V_max,U_min,U_max,inside);

    return inside;
}

template <typename T>
float resample2x2x2(const T             *voxels,
                    const array<ssize_t, 3> &shape,
                    const array<float, 3>   &X) {
    auto  [Nx,Ny,Nz] = shape;

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

namespace NS {

/*
Computes the center of mass of the given tomography.

@param voxels The given tomography.
@returns The 3D coordinates of the center of mass (in Z, Y, X).
*/
array<real_t,3> center_of_mass(const input_ndarray<mask_type> &voxels);

void compute_front_mask(const input_ndarray<mask_type> solid_implant,
        const float voxel_size,
        const matrix4x4 &Muvw,
        std::array<float,6> bbox,
        output_ndarray<mask_type> front_mask);

void cylinder_projection(const input_ndarray<float>  edt,  // Euclidean Distance Transform in um, should be low-resolution (will be interpolated)
             const input_ndarray<uint8_t> C,  // Material classification images (probability per voxel, 0..1 -> 0..255)
             float voxel_size,           // Voxel size for Cs
             float d_min, float d_max,       // Distance shell to map to cylinder
             float theta_min, float theta_max, // Angle range (wrt cylinder center)
             std::array<float,6> bbox,
             const matrix4x4 &Muvw,           // Transform from zyx (in um) to U'V'W' cylinder FoR (in um)
             output_ndarray<float>    image,  // Probability-weighted volume of (class,theta,U)-voxels
             output_ndarray<int64_t>  count   // Number of (class,theta,U)-voxels
             );

void fill_implant_mask(const input_ndarray<mask_type> implant_mask,
               float voxel_size,
               const array<float,6> &bbox,
               float r_fraction,
               const matrix4x4 &Muvw,
               output_ndarray<mask_type> solid_implant_mask,
               output_ndarray<float> rsqr_maxs,
               output_ndarray<float> profile);

/*
Computes the inertia matrix of the given tomography based of the given center of mass.

@param voxels The given tomography.
@param cm The given center of mass.
@returns The 3x3 inertia matrix.
*/
array<real_t,9> inertia_matrix(const input_ndarray<mask_type> &voxels, const array<real_t,3> &cm);

void integrate_axes(const input_ndarray<mask_type> &mask,
		    const array<real_t,3> &x0,
		    const array<real_t,3> &v_axis,
		    const array<real_t,3> &w_axis,
		    const real_t v_min, const real_t w_min,
		    output_ndarray<uint64_t> output);

template <typename T>
void sample_plane(const input_ndarray<T> &voxels,
                  const real_t voxel_size, // In micrometers
                  const array<real_t, 3> cm,
                  const array<real_t, 3> u_axis,
                  const array<real_t, 3> v_axis,
                  const array<real_t, 4>  bbox,    // [umin,umax,vmin,vmax] in micrometers
                  output_ndarray<real_t> plane_samples);

void zero_outside_bbox(const array<real_t,9> &principal_axes,
               const array<real_t,6> &parameter_ranges,
               const array<real_t,3> &cm,
               output_ndarray<mask_type> voxels);
}

#endif