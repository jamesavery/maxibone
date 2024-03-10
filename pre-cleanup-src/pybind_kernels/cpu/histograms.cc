#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <inttypes.h>
#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <tqdm.h>
using namespace std;
namespace py = pybind11;

#include "datatypes.hh"

template <typename voxel_type>
  using np_array = py::array_t<voxel_type, py::array::c_style | py::array::forcecast>;


#define INLINE __attribute__((always_inline)) inline

//#define VALID_VOXEL(voxel) (voxel != 0 && voxel >= vmin && voxel <= vmax) /* Voxel not masked, and within vmin,vmax range */
#define VALID_VOXEL(voxel) (voxel != 0) /* Voxel not masked */



template <typename T> void convolve1d(const np_array<T> np_kernel,
		const np_array<T> np_from,
		np_array<T> &np_to,
		int axis)
{
  auto
    kernel_info = np_kernel.request(),
    from_info   = np_from.request(),
    to_info     = np_kernel.request();

  const T *kernel = static_cast<const T*>(kernel_info.ptr);
  const T *from   = static_cast<const T*>(from_info.ptr);
  T       *to     = static_cast<T*>       (to_info.ptr);


  int64_t
    kernel_size = kernel_info.size,
    padding = kernel_size / 2, // kx should always be odd
    // Partial shape
    Nz = from_info.shape[0],
    Ny = from_info.shape[1],
    Nx = from_info.shape[2];

  assert(Nz == to_info.shape[0] && Ny == to_info.shape[1] && Nx == to_info.shape[2]);

  constexpr size_t n_cache = 65536 / sizeof(T); // Get from OS?
  constexpr size_t n_parallel = n_cache / kernel_size;

  T rolling_buffer[n_cache];
  // i_center = (z_start + padding) % kernel_size
  // update: 1) Load new line into column z_start of rolling_buffer; 2) z_start = (z_start+1) % kernel_size
  // convolve: for(thread = 0; thread < n_parallel; thread++){
  //              T sum = 0;
  //              for(i = 0, z=z_start; i<kernel_size; i++, z = (z+1)%kernel_size) sum += rolling_buffer[thread*kernel_size + z]*kernel[z];

}

template <typename Op, bool neutral> void morphology_3d_sphere_cpu(
        const np_array<mask_type> &np_voxels,
        const int64_t radius,
        const np_array<mask_type> np_result
) {
    auto
        voxels_info = np_voxels.request(),
        result_info = np_result.request();

    int64_t Nz = voxels_info.shape[0], Ny = voxels_info.shape[1], Nx = voxels_info.shape[2];
    int64_t N[3] = {Nz, Ny, Nx};
    int64_t strides[3] = {Ny*Nx, Nx, 1};

    const mask_type *voxels = static_cast<const mask_type*>(voxels_info.ptr);
    mask_type *result = static_cast<mask_type*>(result_info.ptr);

    Op op;

    int64_t sqradius = radius * radius;

    #pragma omp parallel for collapse(3)
    for (int64_t z = 0; z < N[0]; z++) {
        for (int64_t y = 0; y < N[1]; y++) {
            for (int64_t x = 0; x < N[2]; x++) {
                // Compute boundaries
                int64_t flat_index = z*strides[0] + y*strides[1] + x*strides[2];
                int64_t X[3] = {z, y, x};
                int64_t limits[6];
                for (int axis = 0; axis < 3; axis++) {
                    limits[(axis*2)] = -min(radius, X[axis]);
                    limits[(axis*2)+1] = min(radius, N[axis] - X[axis] - 1);
                }

                // Apply the spherical kernel
                bool value = neutral;
                //#pragma omp simd collapse(3) reduction(op:value)
                for (int64_t pz = limits[0]; pz <= limits[1]; pz++) {
                    for (int64_t py = limits[2]; py <= limits[3]; py++) {
                        for (int64_t px = limits[4]; px <= limits[5]; px++) {
                            // TODO exact match with ndimage
                            bool within = px*px + py*py + pz*pz <= sqradius; // sphere kernel
                            int64_t offset = pz*strides[0] + py*strides[1] + px*strides[2];
                            value = within? op(value, voxels[flat_index+offset]) : value;
                        }
                    }
                }

                // Store the results
                result[flat_index] = value;
            }
        }
    }
}

template <typename Op, bool neutral> void morphology_3d_sphere_gpu(
        const np_array<mask_type> &np_voxels,
        const int64_t radius,
        const np_array<mask_type> np_result) {
#ifdef _OPENACC
    auto
        voxels_info = np_voxels.request(),
        result_info = np_result.request();

    int64_t Nz = voxels_info.shape[0], Ny = voxels_info.shape[1], Nx = voxels_info.shape[2];
    int64_t N[3] = {Nz, Ny, Nx};
    int64_t strides[3] = {Ny*Nx, Nx, 1};

    const mask_type *voxels = static_cast<const mask_type*>(voxels_info.ptr);
    mask_type *result = static_cast<mask_type*>(result_info.ptr);

    Op op;
    int64_t sqradius = radius * radius;

    #pragma acc data copyin(voxels[:Nz*Ny*Nx], N[:3], strides[:3], sqradius) copyout(result[:Nz*Ny*Nx])
    {
        #pragma acc parallel loop collapse(3)
        for (int64_t z = 0; z < N[0]; z++) {
            for (int64_t y = 0; y < N[1]; y++) {
                for (int64_t x = 0; x < N[2]; x++) {
                    // Compute boundaries
                    int64_t flat_index = z*strides[0] + y*strides[1] + x*strides[2];
                    int64_t X[3] = {z, y, x};
                    int64_t limits[6];
                    for (int axis = 0; axis < 3; axis++) {
                        limits[(axis*2)] = -min(radius, X[axis]);
                        limits[(axis*2)+1] = min(radius, N[axis] - X[axis] - 1);
                    }

                    // Apply the spherical kernel
                    bool value = neutral;
                    //#pragma omp simd collapse(3) reduction(op:value)
                    for (int64_t pz = limits[0]; pz <= limits[1]; pz++) {
                        for (int64_t py = limits[2]; py <= limits[3]; py++) {
                            for (int64_t px = limits[4]; px <= limits[5]; px++) {
                                bool within = px*px + py*py + pz*pz <= sqradius; // sphere kernel
                                int64_t offset = pz*strides[0] + py*strides[1] + px*strides[2];
                                value = within? op(value, voxels[flat_index+offset]) : value;
                            }
                        }
                    }

                    // Store the results
                    result[flat_index] = value;
                }
            }
        }
    }
#else
    throw runtime_error("Library wasn't compiled with OpenACC.");
#endif
}

pair<float,float> float_minmax(const np_array<float> np_field) {
    // Extract NumPy array basearray-pointer and length
    auto field_info    = np_field.request();
    size_t image_length = field_info.size;
    const float *field = static_cast<const float*>(field_info.ptr);

    float voxel_min = field[0], voxel_max = field[0];

    #pragma omp parallel for reduction(min:voxel_min) reduction(max:voxel_max)
    for (size_t i=0; i < image_length; i++) {
      float value = field[i];
      voxel_min = min(voxel_min, value);
      voxel_max = max(voxel_max, value);
    }

    return make_pair(voxel_min,voxel_max);
}

bool in_bbox(float U, float V, float W, const std::array<float,6> bbox)
{
  const auto& [U_min,U_max,V_min,V_max,W_min,W_max] = bbox;

  return U>=U_min && U<=U_max && V>=V_min && V<=V_min && W>=W_min && W<=W_max;
}


INLINE
float resample2x2x2(const field_type      *voxels,
		    const array<ssize_t,3> &shape,
		    const array<float,3>   &X)
{
  auto  [Nx,Ny,Nz] = shape;	// Eller omvendt?

  if(!in_bbox(X[0],X[1],X[2], {0.5,Nx-1.5, 0.5,Ny-1.5, 0.5,Nz-1.5})){
    uint64_t voxel_index = floor(X[0])*Ny*Nz+floor(X[1])*Ny+floor(X[2]);
    return voxels[voxel_index];
  }
  float   Xfrac[2][3];	// {Xminus[3], Xplus[3]}
  int64_t  Xint[2][3];	// {Iminus[3], Iplus[3]}
  float   value = 0;

  for(int i=0;i<3;i++){
    double Iminus, Iplus;
    Xfrac[0][i] = 1-modf(X[i]-0.5, &Iminus); // 1-{X[i]-1/2}, floor(X[i]-1/2)
    Xfrac[1][i] =   modf(X[i]+0.5, &Iplus);  // {X[i]+1/2}, floor(X[i]+1/2)

    Xint[0][i] = Iminus;
    Xint[1][i] = Iplus;
  }


  for(int ijk=0; ijk<=7; ijk++) {
    float  weight = 1;
    int64_t IJK[3] = {0,0,0};

    for(int axis=0;axis<3;axis++){ // x-1/2 or x+1/2
      int pm = (ijk>>axis) & 1;
      IJK[axis] = Xint[pm][axis];
      weight   *= Xfrac[pm][axis];
    }

    auto [I,J,K] = IJK;
    if(I<0 || J<0 || K<0){
      printf("(I,J,K) = (%ld,%ld,%ld)\n",I,J,K);

      abort();
    }
    if(I>=int(Nx) || J>=int(Ny) || K>=int(Nz)){
      printf("(I,J,K) = (%ld,%ld,%ld), (Nx,Ny,Nz) = (%ld,%ld,%ld)\n",I,J,K,Nx,Ny,Nz);
      abort();
    }
    uint64_t voxel_index = I*Ny*Nz+J*Ny+K;
    field_type voxel = voxels[voxel_index];
    value += voxel*weight;
  }
  return value;
}

void otsu(
        const np_array<uint64_t> np_bins,
        np_array<uint64_t> np_result,
        uint64_t step_size) {
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
            #pragma omp simd reduction(+:w0)
            for (uint64_t j = 0; j < th; j++) {
                w0 += row[j];
            }

            // w1 = row[th:].sum()
            #pragma omp simd reduction(+:w1)
            for (uint64_t j = th; j < N_cols; j++) {
                w1 += row[j];
            }
            float_t fw0 = 1.0f / w0, fw1 = 1.0f / w1;

            //if w0 <= 0 or w1 <= 0:
            //    return np.inf
            if (w0 == 0 || w1 == 0) {
                guesses[guess] = (uint64_t) -1;
            } else {
                float_t m0 = 0, m1 = 0;
                // m0 = (1/w0) * (np.arange(th)*row[:th]).sum()
                #pragma omp simd reduction(+:m0)
                for (uint64_t j = 0; j < th; j++) {
                    m0 += j * row[j];
                }
                m0 *= fw0;

                // m1 = (1/w1) * (np.arange(row.shape[0]-th)*row[th:]).sum()
                #pragma omp simd reduction(+:m1)
                for (uint64_t j = th; j < N_cols; j++) {
                    m1 += j * row[j];
                }
                m1 *= fw1;


                float_t s0 = 0, s1 = 0;
                // s0 = (1/w0) * (((np.arange(th)-m0)**2)*row[:th]).sum()
                #pragma omp simd reduction(+:s0)
                for (uint64_t j = 0; j < th; j++) {
                    uint64_t im0 = j - m0;
                    s0 += (im0*im0) * row[j];
                }
                s0 *= fw0;

                // s1 = (1/w1) * (((np.arange(row.shape[0]-th)-m1)**2)*row[th:]).sum()
                #pragma omp simd reduction(+:s1)
                for (uint64_t j = th; j < N_cols; j++) {
                    uint64_t im1 = j - m1;
                    s1 += (im1*im1) * row[j];
                }
                s1 *= fw1;

                // return w0*s0 + w1*s1
                guesses[guess] = (uint64_t) floor(w0*s0 + w1*s1);
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

PYBIND11_MODULE(histograms, m) {
    m.doc() = "2D histogramming plugin"; // optional module docstring
    m.def("float_minmax", &float_minmax);
    m.def("otsu", &otsu);
}
