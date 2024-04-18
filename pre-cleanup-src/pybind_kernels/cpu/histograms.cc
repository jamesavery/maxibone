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

PYBIND11_MODULE(histograms, m) {
    m.doc() = "2D histogramming plugin"; // optional module docstring
    m.def("float_minmax", &float_minmax);
}
