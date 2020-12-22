#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <inttypes.h>
#include <stdio.h>
namespace py = pybind11;

typedef uint8_t voxel_type;
typedef float   field_type;

// On entry, x_bins[Nx*voxel_bins], y_bins[Ny*voxel_bins], z_bins[Nz*voxel_bins], r_bins[Nr*voxel_bins] must be allocated and zeroed
void axis_histogram(const py::array_t<voxel_type> np_voxels,
		    py::array_t<uint64_t> &np_x_bins,
		    py::array_t<uint64_t> &np_y_bins,
		    py::array_t<uint64_t> &np_z_bins,
		    py::array_t<uint64_t> &np_r_bins,		    
		    const double vmin, const double vmax)
{
  py::buffer_info
    voxels_info = np_voxels.request(),
    x_info = np_x_bins.request(),
    y_info = np_y_bins.request(),
    z_info = np_z_bins.request(),
    r_info = np_r_bins.request();
  
  const uint64_t
    image_length = voxels_info.size,
    voxel_bins   = x_info.shape[1],
    Nx = x_info.shape[0],
    Ny = y_info.shape[0],
    Nz = z_info.shape[0],
    Nr = r_info.shape[0];

  uint64_t
    *x_bins = static_cast<uint64_t*>(x_info.ptr),
    *y_bins = static_cast<uint64_t*>(y_info.ptr),    
    *z_bins = static_cast<uint64_t*>(z_info.ptr),
    *r_bins = static_cast<uint64_t*>(r_info.ptr);

  const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);
  
  uint64_t i=0;

  fprintf(stderr,"Starting binning: (vmin,vmax) = (%g,%g), (Nx,Ny,Nz,Nr) = (%ld,%ld,%ld,%ld)\n",vmin, vmax, Nx,Ny,Nz,Nr);

#pragma omp parallel for private(i) reduction(+:x_bins[:Nx*voxel_bins], y_bins[:Ny*voxel_bins], z_bins[:Nz*voxel_bins], r_bins[:Nr*voxel_bins])
  for(i=0;i< image_length;i++)
    if(voxels[i] != 0) {	// 0 is out-of-band value for masked voxels; voxel values start at 1
      uint64_t x = i % Nx;
      uint64_t y = (i / Nx) % Ny;
      uint64_t z = i / (Nx*Ny);
      uint64_t r = round(sqrt((x-Nx/2.0)*(x-Nx/2.0) + (y-Ny/2.0)*(y-Ny/2.0)));
      
      int64_t voxel_index = round((voxel_bins-1) * (voxels[i] - vmin)/(vmax - vmin) );
      
      // fprintf(stderr,
      // 	   "x %ld %ld: %ld < %ld\n"
      // 	   "y %ld %ld: %ld < %ld\n"
      // 	   "z %ld %ld: %ld < %ld\n"
      // 	   "r %ld %ld: %ld < %ld\n",
      // 	   x, voxel_index, x*voxel_bins + voxel_index, Nx*voxel_bins,
      // 	   y, voxel_index, y*voxel_bins + voxel_index, Ny*voxel_bins,
      // 	   z, voxel_index, z*voxel_bins + voxel_index, Nz*voxel_bins,
      // 	   r, voxel_index, r*voxel_bins + voxel_index, Nr*voxel_bins);

      x_bins[x*voxel_bins + voxel_index]++;
      y_bins[y*voxel_bins + voxel_index]++;    
      z_bins[z*voxel_bins + voxel_index]++;    
      r_bins[r*voxel_bins + voxel_index]++;

  }
}

void field_histogram(const py::array_t<voxel_type> np_voxels,
		     const py::array_t<field_type> np_field,
		     py::array_t<uint64_t> &np_bins,		     
		     const double vmin, const double vmax,
		     const double fmin, const double fmax)
{
  py::buffer_info
    voxels_info = np_voxels.request(),
    field_info = np_field.request(),
    bins_info = np_bins.request();
  
  const uint64_t
    image_length = voxels_info.size,
    field_bins   = bins_info.shape[0],
    voxel_bins   = bins_info.shape[1];

  uint64_t *bins = static_cast<uint64_t*>(bins_info.ptr);

  const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);
  const field_type *field  = static_cast<field_type*>(field_info.ptr);  

  uint64_t i=0;

#pragma omp parallel for private(i) reduction(+:bins[:field_bins*voxel_bins])  
  for(i=0;i< image_length;i++)
    if(voxels[i] != 0) {	// 0 is special value for masked voxels; voxel values start at 1
      int64_t voxel_index = round((voxel_bins-1) * (voxels[i] - vmin)/(vmax - vmin) );
      int64_t field_index = round((field_bins-1) * (field[i] - fmin)/(fmax - fmin) );      

      bins[field_index*voxel_bins + voxel_index]++;
    }
}

PYBIND11_MODULE(histograms, m) {
    m.doc() = "2D histogramming plugin"; // optional module docstring

    m.def("axis_histogram",  &axis_histogram, "this is how you do it");
    m.def("field_histogram", &field_histogram, "this is how you do it");
}
