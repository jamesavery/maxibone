#ifndef geometry_h
#define geometry_h

using namespace std;

#include "datatypes.hh"
#include <chrono>
#include <string>

#define dot(a,b) (a[0]*b[0] + a[1]*b[1] + a[2]*b[2])

void print_timestamp(string message) {
    auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    tm local_tm = *localtime(&now);
    fprintf(stderr,"%s at %02d:%02d:%02d\n", message.c_str(), local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
}

namespace NS {

/*
Computes the center of mass of the given tomography.

@param voxels The given tomography.
@returns The 3D coordinates of the center of mass (in Z, Y, X).
*/
array<real_t,3> center_of_mass(const input_ndarray<mask_type> &voxels);

bool in_bbox(float U, float V, float W, const std::array<float, 6> &bbox);

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
		    output_ndarray<real_t> output);

template <typename T>
float resample2x2x2(const T *voxels,
                    const array<ssize_t,3> &shape,
                    const array<float,3> &X);

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