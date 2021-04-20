// TODO: OpenACC
array<double,3> center_of_mass(const voxel_type *voxels, const size_t image_length) {
  array<double,3> cm = {0,0,0};

#pragma omp parallel for reduction(+:cm[:3])
  for(size_t flat_idx=0;flat_idx<image_length;flat_idx++){
    uint64_t x = flat_idx % Nx;
    uint64_t y = (flat_idx / Nx) % Ny;
    uint64_t z = flat_idx / (Nx*Ny);

    double m = voxels[flat_idx];
		      
    cm[0] += m*x; cm[1] += m*y; cm[2] += m*z;
  }

  return cm;
}

array<double,9> inertia_matrix(const voxel_types *voxels, const size_t image_length, const array<double,3> &cm)
{
  array<double,9> M = {0,0,0,
		       0,0,0,
		       0,0,0};

#pragma omp parallel for reduction(+:M[:9]) 
  for(size_t flat_idx=0;flat_idx<image_length;flat_idx++)
    if(voxels[flat_idx] != 0) { // TODO: Check if faster with or without test
      // x,y,z
      uint64_t xs[3] = {flat_idx % Nx, (flat_idx / Nx) % Ny, flat_idx / (Nx*Ny)};
      
      for(int i=0;i<3;i++){
	M[i,i] += m[flat_idx] * (xs[0]*xs[0] + xs[1]*xs[1] + xs[2]*xs[2]);
	for(int j=0;j<3;j++)
	  M[i*3 + j] -= m[flat_idx] * xs[i] * xs[j];
      }
    }

  return M;
}

namespace python_api {
  
  array<double,3> center_of_mass(const py::array_t<voxel_type> &np_voxels){
    auto voxels_info    = np_voxels.request();
    size_t image_length = voxels_info.size;
    const voxel_type *voxels = static_cast<const voxel_type*>(voxels_info.ptr);

    return center_of_mass(voxels,mage_length);
  }


  array<double,9> inertia_matrix(const py::array_t<voxel_type> &np_voxels){
    auto voxels_info    = np_voxels.request();
    size_t image_length = voxels_info.size;
    const voxel_type *voxels = static_cast<const voxel_type*>(voxels_info.ptr);

    array<double,3> cm = center_of_mass(voxels,image_length);
    
    return center_of_mass(voxels,mage_length, cm);
  }
  
}
