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
