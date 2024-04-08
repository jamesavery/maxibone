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

inline bool in_bbox(const std::array<float, 3> index, const std::array<float, 6> &bbox) {
    const auto& [z, y, x] = index;
    const auto& [zmin, zmax, ymin, ymax, xmin, xmax] = bbox;

    return
        z >= zmin && z <= zmax &&
        y >= ymin && y <= ymax &&
        x >= xmin && x <= xmax;
}

template <typename T>
float resample2x2x2(const T                      *voxels,
                    const std::array<ssize_t, 3> &shape,
                    const std::array<float, 3>   &index) {
    auto  [Nz,Ny,Nx] = shape;

    if (!in_bbox(index, {0.5f, float(Nx)-0.5f, 0.5f, float(Ny)-0.5f, 0.5f, float(Nz)-0.5f})) {
        uint64_t voxel_index = uint64_t(floor(index[0]))*Ny*Nx + uint64_t(floor(index[1]))*Nx + uint64_t(floor(index[2]));
        return voxels[voxel_index];
    }

    float   Ifrac[2][3]; // {Xminus[3], Xplus[3]}
    int64_t Iint[2][3];  // {Iminus[3], Iplus[3]}
    float   value = 0;

    for (int i = 0; i < 3; i++) {
        float Iminus, Iplus;
        Ifrac[0][i] = 1-std::modf(index[i]-0.5f, &Iminus); // 1-{X[i]-1/2}, floor(X[i]-1/2)
        Ifrac[1][i] =   std::modf(index[i]+0.5f, &Iplus);  // {X[i]+1/2}, floor(X[i]+1/2)

        Iint[0][i] = (int64_t) Iminus;
        Iint[1][i] = (int64_t) Iplus;
    }

    for (int ijk = 0; ijk <= 7; ijk++) {
        float  weight = 1;
        int64_t IJK[3] = {0,0,0};

        for (int axis = 0; axis < 3; axis++) { // x-1/2 or x+1/2
            int pm    = (ijk >> axis) & 1;
            IJK[axis] = Iint[pm][axis];
            weight   *= Ifrac[pm][axis];
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
        uint64_t voxel_index = I*Ny*Nx + J*Nx + K;
        //assert(I>=0 && J>=0 && K>=0);
        //assert(I<Nx && J<Ny && K<Nz);
        float voxel = (float) voxels[voxel_index];
        value += voxel * weight;
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

void fill_implant_mask_pre(const input_ndarray<mask_type> mask,
               int64_t offset,
               float voxel_size,
               const array<float,6> &bbox,
               float r_fraction,
               const matrix4x4 &Muvw,
               output_ndarray<real_t> thetas,
               output_ndarray<float> rsqr_maxes);

void fill_implant_mask(const input_ndarray<mask_type> mask,
               int64_t offset,
               float voxel_size,
               const array<float,6> &bbox,
               float r_fraction,
               const matrix4x4 &Muvw,
               const input_ndarray<real_t> thetas,
               const input_ndarray<float> rsqr_maxs,
               output_ndarray<mask_type> solid_implant_mask,
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