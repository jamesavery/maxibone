#include <array>
#include <chrono>
#include <inttypes.h>
#include <stdio.h>


using namespace std;
typedef uint8_t voxel_type;

void print_timestamp(string message)
{
  auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
  tm local_tm = *localtime(&now);
  fprintf(stderr,"%s at %02d:%02d:%02d\n", message.c_str(), local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);    
}

// TODO: OpenACC
array<double,3> center_of_mass(const voxel_type *voxels, const array<size_t,3> &shape) {
  //  double cm[3] = {0,0,0};
  double cmx = 0, cmy = 0, cmz = 0;
  size_t  Nz = shape[0], Ny = shape[1], Nx = shape[2];
  int64_t image_length = Nx*Ny*Nz;
  int64_t block_size =  1024 * 1024 * 1024/sizeof(voxel_type); // 1 GB

  print_timestamp("center_of_mass start");
  //#pragma omp parallel for reduction(+:cm[:3]) simd


  
  for(int64_t block_start=0;block_start<image_length;block_start+=block_size){

    const voxel_type *buffer = voxels + block_start;
    int64_t     block_length = min(block_size,image_length-block_start);

#pragma acc parallel loop reduction(+:cmx,cmy,cmz) copyin(buffer[:block_length])    
    for(uint64_t k = 0; k<block_length;k++){
      uint64_t flat_idx = block_start + k;
      uint64_t x = flat_idx % Nx;
      uint64_t y = (flat_idx / Nx) % Ny;
      uint64_t z = flat_idx / (Nx*Ny);
      
      double m = buffer[k];
    
    //    cm[0] += m*x; cm[1] += m*y; cm[2] += m*z;
      cmx += m*x; cmy += m*y; cmz += m*z;
    }
  }

  print_timestamp("center_of_mass end");  

  return array<double,3>{cmx,cmy,cmz};
}

array<double,9> inertia_matrix_reduction(const voxel_type *voxels, const array<size_t,3> &shape, const array<double,3> &cm)
{
  double
    M00 = 0, M01 = 0, M02 = 0,
    M10 = 0, M11 = 0, M12 = 0,
    M20 = 0, M21 = 0, M22 = 0;
  
  size_t Nz = shape[0], Ny = shape[1], Nx = shape[2];
  size_t image_length = Nx*Ny*Nz;
  size_t block_size =  1024 * 1024 * 1024/sizeof(voxel_type); // 1 GB

  print_timestamp("inertia_matrix start");    
  
  for(size_t block_start=0;block_start<image_length;block_start+=block_size){
    const voxel_type *buffer  = voxels + block_start;
    int block_length = min(block_size,image_length-block_start);

#pragma acc parallel loop copyin(buffer[:block_length]) reduction(+:M00,M01,M02,M10,M11,M12,M20,M21,M22)
    for(uint64_t k = 0; k<block_length;k++)  if(buffer[k] != 0) {    
	uint64_t flat_idx = block_start + k;
	uint64_t xs[3] = {flat_idx % Nx, (flat_idx / Nx) % Ny, flat_idx / (Nx*Ny)}; // x,y,z

	double diag = voxels[flat_idx] * (xs[0]*xs[0] + xs[1]*xs[1] + xs[2]*xs[2]);
	M00 += diag - voxels[flat_idx] * xs[0] * xs[0];
	M11 += diag - voxels[flat_idx] * xs[1] * xs[1];
	M22 += diag - voxels[flat_idx] * xs[2] * xs[2];	
	M01 -= voxels[flat_idx] * xs[0] * xs[1];
	M10 -= voxels[flat_idx] * xs[0] * xs[1];
	M02 -= voxels[flat_idx] * xs[0] * xs[2];
	M01 -= voxels[flat_idx] * xs[0] * xs[2];
	M12 -= voxels[flat_idx] * xs[1] * xs[2];
	M21 -= voxels[flat_idx] * xs[1] * xs[2]; 			
      }
  }
  print_timestamp("inertia_matrix end");      
  return array<double,9> {M00,M01,M02,M10,M11,M12,M20,M21,M22};
}


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


namespace python_api { 
  namespace py = pybind11;
  
  
  array<double,3> center_of_mass(const py::array_t<voxel_type, py::array::c_style | py::array::forcecast> np_voxels){
    auto voxels_info    = np_voxels.request();
    const voxel_type *voxels = static_cast<const voxel_type*>(voxels_info.ptr);
    array<size_t,3> shape = {voxels_info.shape[0],voxels_info.shape[1],voxels_info.shape[2]};        

    return ::center_of_mass(voxels,shape);
  }


  array<double,9> inertia_matrix(const py::array_t<voxel_type, py::array::c_style | py::array::forcecast> np_voxels){
    auto voxels_info    = np_voxels.request();
    const voxel_type *voxels = static_cast<const voxel_type*>(voxels_info.ptr);
    array<size_t,3> shape = {voxels_info.shape[0],voxels_info.shape[1],voxels_info.shape[2]};    
    
    array<double,3> cm = ::center_of_mass(voxels, shape);
    return ::inertia_matrix_reduction(voxels,shape, cm);
  }

}



PYBIND11_MODULE(geometry, m) {
    m.doc() = "Voxel Geometry Module"; // optional module docstring

    m.def("center_of_mass",  &python_api::center_of_mass);
    m.def("inertia_matrix",  &python_api::inertia_matrix);
}
