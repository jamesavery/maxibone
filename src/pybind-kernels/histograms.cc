#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <inttypes.h>
#include <stdio.h>
#include <omp.h>
using namespace std;
namespace py = pybind11;

typedef uint16_t voxel_type;
typedef float    field_type;

std::pair<int,int> masked_minmax(const py::array_t<voxel_type> np_voxels)
{
  // Extract NumPy array basearray-pointer and length
  auto voxels_info    = np_voxels.request();
  size_t image_length = voxels_info.size;
  const voxel_type *voxels = static_cast<const voxel_type*>(voxels_info.ptr);
  
  voxel_type voxel_min = std::max(voxels[0], voxel_type(1)), voxel_max = voxels[0];
  size_t i=0;

#pragma omp parallel for private(i) reduction(min:voxel_min) reduction(max:voxel_max)
  for(i=0;i<image_length;i++){
    voxel_min = std::min(voxel_min, voxels[i]>0? voxels[i]:voxel_type(1));
    voxel_max = std::max(voxel_max, voxels[i]);
  }

  assert(voxel_min > 0);
  return std::make_pair(voxel_min,voxel_max);
}

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

  // uint64_t
  //   *x_bins = static_cast<uint64_t*>(x_info.ptr),
  //   *y_bins = static_cast<uint64_t*>(y_info.ptr),    
  //   *z_bins = static_cast<uint64_t*>(z_info.ptr),
  //   *r_bins = static_cast<uint64_t*>(r_info.ptr);

  const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);

  fprintf(stderr,"\n\nStarting winning %p: (vmin,vmax) = (%g,%g), (Nx,Ny,Nz,Nr) = (%ld,%ld,%ld,%ld)\n",voxels,vmin, vmax, Nx,Ny,Nz,Nr);
  uint64_t i=0;
  int64_t voxel_max = 0;

  fprintf(stderr,"Allocating memory (%ld bytes)\n", (Nx+Ny+Nz+Nr)*voxel_bins*8);
  uint64_t *x_bins = (uint64_t*)calloc(Nx*voxel_bins,8), *y_bins = (uint64_t*)calloc(Ny*voxel_bins,8),
           *z_bins = (uint64_t*)calloc(Nz*voxel_bins,8), *r_bins = (uint64_t*)calloc(Nr*voxel_bins,8);

  fprintf(stderr,"Starting calculation\n\n");

  // TODO: Change stack size to accommodate reduction arrays (160MB ish)
  //#pragma omp parallel for private(i,voxel_max) reduction(+:x_bins[:Nx*voxel_bins], y_bins[:Ny*voxel_bins], z_bins[:Nz*voxel_bins], r_bins[:Nr*voxel_bins])
  for(i=0;i< image_length;i++){
    uint64_t x = i % Nx;
    uint64_t y = (i / Nx) % Ny;
    uint64_t z = i / (Nx*Ny);
    uint64_t r = floor(sqrt((x-Nx/2.0)*(x-Nx/2.0) + (y-Ny/2.0)*(y-Ny/2.0)));
      
    int64_t voxel_index = floor(static_cast<double>(voxel_bins-1) * ((voxels[i] - vmin)/(vmax - vmin)) );    
    
    voxel_max = std::max(voxel_index,voxel_max);

    if(x >= Nx || y >= Ny || z >= Nz || r >= Nr || voxel_index >= (int64_t)voxel_bins){
      fprintf(stderr,"Out-of-bounds error for index %ld:\n"
    	   "x %ld < %ld;  %ld: %ld < %ld\n"
    	   "y %ld < %ld;  %ld: %ld < %ld\n"
    	   "z %ld < %ld;  %ld: %ld < %ld\n"
    	   "r %ld < %ld;  %ld: %ld < %ld\n",
	      i,
	      x, Nx, voxel_index, x*voxel_bins + voxel_index, Nx*voxel_bins,
	      y, Ny, voxel_index, y*voxel_bins + voxel_index, Ny*voxel_bins,
	      z, Nz, voxel_index, z*voxel_bins + voxel_index, Nz*voxel_bins,
	      r, Nr, voxel_index, r*voxel_bins + voxel_index, Nr*voxel_bins);
    } else if(voxels[i] != 0) {	
      x_bins[x*voxel_bins + voxel_index]++;
      y_bins[y*voxel_bins + voxel_index]++;    
      z_bins[z*voxel_bins + voxel_index]++;    
      r_bins[r*voxel_bins + voxel_index]++;
    }
  }
  fprintf(stderr,"voxel_max = %ld\n",voxel_max);

  memcpy(x_info.ptr, x_bins, Nx*voxel_bins*sizeof(uint64_t));
  memcpy(y_info.ptr, y_bins, Ny*voxel_bins*sizeof(uint64_t));
  memcpy(z_info.ptr, z_bins, Nz*voxel_bins*sizeof(uint64_t));
  memcpy(r_info.ptr, r_bins, Nr*voxel_bins*sizeof(uint64_t));
  free(x_bins);
  free(y_bins);
  free(z_bins);
  free(r_bins);
}

// TODO: Allow field to be lower resolution than voxel data
void field_histogram(const py::array_t<voxel_type> np_voxels,
		     const py::array_t<field_type> np_field,
		     py::array_t<uint64_t> &np_bins,		     
		     const double vmin, const double vmax
	     )
{
  py::buffer_info
    voxels_info = np_voxels.request(),
    field_info = np_field.request(),
    bins_info = np_bins.request();
  
  const uint64_t
    image_length = voxels_info.size,
    field_length = field_info.size,    
    field_bins   = bins_info.shape[0],
    voxel_bins   = bins_info.shape[1];

  const uint64_t
    nZ = voxels_info.shape[0], nY = voxels_info.shape[1], nX = voxels_info.shape[2],
    nz = field_info.shape[0],  ny = field_info.shape[1],  nx = field_info.shape[2];

  double dz = nz/((double)nZ), dy = ny/((double)nY), dx = nx/((double)nX);
  
  const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);
  const field_type *field  = static_cast<field_type*>(field_info.ptr);  
  uint64_t *bins = static_cast<uint64_t*>(bins_info.ptr);
  
  uint64_t I=0;

  float fmin=1e6, fmax=0;	// TODO: inf, -inf
  for(uint64_t i=0;i<field_length;i++){
    fmin = field[i]>0? min(field[i],fmin) : fmin; // TODO: Should we really mask field zeros?
    fmax = max(field[i],fmax);
  }
  
  //#pragma omp parallel for private(I) reduction(+:bins[:field_bins*voxel_bins])  
  for(I=0;I<image_length;I++){
    int64_t voxel_index = floor(static_cast<double>(voxel_bins-1) * ((voxels[I] - vmin)/(vmax - vmin)) );

    // What are the X,Y,Z indices corresponding to voxel basearray index I?
    uint64_t X = I % nX, Y = (I / nX) % nY, Z = I / (nX*nY);

    // And what are the corresponding x,y,z coordinates into the field array, and field basearray index i?
    // TODO: Sample 2x2x2 volume?
    uint64_t x = floor(X*dx), y = floor(Y*dy), z = floor(Z*dz);
    uint64_t i = z*ny*nx + y*nx + x;

    if((voxels[I]>=1) && (field[i]>0)){ // Mask zeros in both voxels and field (TODO: should field be masked, or 0 allowed?)
      int64_t field_index = floor(static_cast<double>(field_bins-1) * ((field[i] - fmin)/(fmax - fmin)) );
      
      bins[field_index*voxel_bins + voxel_index]++;
    }
  }
}

PYBIND11_MODULE(histograms, m) {
    m.doc() = "2D histogramming plugin"; // optional module docstring

    m.def("axis_histogram",  &axis_histogram);
    m.def("field_histogram", &field_histogram);
    m.def("masked_minmax", &masked_minmax);
}
