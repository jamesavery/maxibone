#include <chrono>
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <math.h>
using namespace std;

#include "datatypes.hh"

#define dot(a,b) (a[0]*b[0] + a[1]*b[1] + a[2]*b[2])

void print_timestamp(string message)
{
  auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
  tm local_tm = *localtime(&now);
  fprintf(stderr,"%s at %02d:%02d:%02d\n", message.c_str(), local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);    
}


array<real_t,3> center_of_mass(const ndarray_input<voxel_type> voxels) {
  // nvc++ doesn't support OpenACC 2.7 array reductions yet.  
  real_t cmx = 0, cmy = 0, cmz = 0;
  size_t  Nz = voxels.shape[0], Ny = voxels.shape[1], Nx = voxels.shape[2];
  int64_t image_length = Nx*Ny*Nz;

  print_timestamp("center_of_mass start");
  real_t total_mass = 0;  
  for(int64_t block_start=0;block_start<image_length;block_start+=acc_block_size){

    const voxel_type *buffer = voxels.data + block_start;
    ssize_t this_block_length = min(acc_block_size,image_length-block_start);

#pragma acc parallel loop reduction(+:cmx,cmy,cmz,total_mass) copyin(buffer[:this_block_length])
    for(uint64_t k = 0; k<this_block_length;k++){
      real_t          m = buffer[k];      

      uint64_t flat_idx = block_start + k;
      uint64_t x = flat_idx % Nx;
      uint64_t y = (flat_idx / Nx) % Ny;
      uint64_t z = flat_idx / (Nx*Ny);

      total_mass += m;
      cmx += m*x; cmy += m*y; cmz += m*z;
    }
  }
  cmx /= total_mass; cmy /= total_mass; cmz /= total_mass;
  
  print_timestamp("center_of_mass end");  

  return array<real_t,3>{cmx,cmy,cmz};
}


array<real_t,9> inertia_matrix_serial(const ndarray_input<voxel_type> &voxels, const array<real_t,3> &cm)
{
  real_t
    Ixx = 0, Ixy = 0, Ixz = 0,
             Iyy = 0, Iyz = 0,
                      Izz = 0;
  
  size_t Nz = voxels.shape[0], Ny = voxels.shape[1], Nx = voxels.shape[2];

  print_timestamp("inertia_matrix_serial start");
  for(uint64_t Z=0,k=0;Z<Nz;Z++)
    for(uint64_t Y=0;Y<Ny;Y++)
      for(uint64_t X=0;X<Nx;X++,k++){
	real_t x = X-cm[0], y = Y-cm[1], z = Z-cm[2];
	
	real_t m = voxels.data[k];
	Ixx += m*(y*y+z*z);
	Iyy += m*(x*x+z*z);
	Izz += m*(x*x+y*y);	
	Ixy -= m * x*y;
	Ixz -= m * x*z;
	Iyz -= m * y*z;
      }
  
  print_timestamp("inertia_matrix_serial end");      
  return array<real_t,9> {
    Ixx, Ixy, Ixz,
    Ixy, Iyy, Iyz,
    Ixz, Iyz, Izz
  };
}


array<real_t,9> inertia_matrix(const ndarray_input<voxel_type> &voxels, const array<real_t,3> &cm)
{
  // nvc++ doesn't support OpenACC 2.7 array reductions yet, so must name each element.
  real_t
    M00 = 0, M01 = 0, M02 = 0,
             M11 = 0, M12 = 0,
                      M22 = 0;
  
  size_t Nz = voxels.shape[0], Ny = voxels.shape[1], Nx = voxels.shape[2];
  ssize_t image_length = Nx*Ny*Nz;

  print_timestamp("inertia_matrix start");    
  for(ssize_t block_start=0;block_start<image_length;block_start+=acc_block_size){
    const voxel_type *buffer  = voxels.data + block_start;
    ssize_t block_length = min(acc_block_size,image_length-block_start);

#ifdef _OPENACC
#pragma acc parallel loop copyin(buffer[:block_length]) reduction(+:M00,M01,M02,M11,M12,M22)
#else
#pragma omp parallel for reduction(+:M00,M01,M02,M11,M12,M22)
#endif    
    for(uint64_t k = 0; k<block_length;k++) {    //\if(buffer[k] != 0)
	uint64_t flat_idx = block_start + k;
	real_t xs[3] = {(flat_idx % Nx)        - cm[0],  // x
			((flat_idx / Nx) % Ny) - cm[1],  // y
			(flat_idx / (Nx*Ny))   - cm[2]}; // z

	real_t m = buffer[k];
	real_t diag = dot(xs,xs);
	M00 += m*(diag - xs[0] * xs[0]);
	M11 += m*(diag - xs[1] * xs[1]);
	M22 += m*(diag - xs[2] * xs[2]);	
	M01 -= m * xs[0] * xs[1];
	M02 -= m * xs[0] * xs[2];
	M12 -= m * xs[1] * xs[2];
      }
  }
  print_timestamp("inertia_matrix end");      
  return array<real_t,9> {
    M00,M01,M02,
    M01,M11,M12,
    M02,M12,M22};
}


void integrate_axes(const ndarray_input<voxel_type> &voxels,
		    const array<real_t,3> &x0,		    
		    const array<real_t,3> &v_axis,
		    const array<real_t,3> &w_axis,
		    const real_t v_min, const real_t w_min,
		    ndarray_output<real_t> output)
{
  size_t Nz = voxels.shape[0], Ny = voxels.shape[1], Nx = voxels.shape[2];
  size_t Nv = output.shape[0], Nw = output.shape[1]; 
  int64_t image_length = Nx*Ny*Nz;
  
  for(ssize_t block_start=0;block_start<image_length;block_start += acc_block_size){
    const voxel_type *buffer  = voxels.data + block_start;
    int block_length = min(acc_block_size,image_length-block_start);

    //#pragma acc parallel loop copy(output.data[:Nv*Nw]) copyin(buffer[:block_length], x0, v_axis, w_axis) 
    for(uint64_t k = 0; k<block_length;k++) { // if(buffer[k] != 0) {
	uint64_t flat_idx = block_start + k;
	real_t xs[3] = {(flat_idx % Nx)        - x0[0],  // x
			((flat_idx / Nx) % Ny) - x0[1],  // y
			(flat_idx / (Nx*Ny))   - x0[2]}; // z


	voxel_type voxel = buffer[k];
	real_t v = dot(xs,v_axis), w = dot(xs,w_axis);
	int64_t i_v = round(v-v_min), j_w = round(w-w_min);

	if(i_v < 0 || j_w < 0 || i_v >= Nv || j_w >= Nw){
	  fprintf(stderr,"(x,y,z) = (%g,%g,%g), (v,w) = (%g,%g), (i,j) = (%ld,%ld)\n",
		  xs[0],xs[1],xs[2],
		  v,w,
		  i_v, j_w);
	  abort();
	}

	if(i_v >= 0 && j_w >= 0 && i_v < Nv && j_w < Nw){
          #pragma acc atomic
	  output.data[i_v*Nw + j_w] += voxel;
	}
      }
  }
}


void sample_plane(const ndarray_input<voxel_type> &voxels, const plane_t &plane, ndarray_output<voxel_type> plane_samples, array<real_t,3> L)
{
  ssize_t Ny = voxels.shape[1], Nx = voxels.shape[2];
  ssize_t nu = plane_samples.shape[0], nv = plane_samples.shape[1];  
  real_t  du = L[0]/nu, dv = L[1]/nv;

  for(ssize_t ui=0;ui<nu;ui++)
    for(ssize_t vj=0;vj<nv;vj++){
      real_t u = (ui-nu/2)*du, v = (vj-nv/2)*dv;

      real_t x = plane.cm[0] + u*plane.u_axis[0] + v*plane.v_axis[0], y = plane.cm[1] + u*plane.u_axis[1] + v*plane.v_axis[1], z = plane.cm[2] + u*plane.u_axis[2] + v*plane.v_axis[2];
      real_t   X[3] = {x,y,z};
      real_t   Xfrac[2][3];	// {Xminus[3], Xplus[3]}
      uint64_t Xint[2][3];	// {Iminus[3], Iplus[3]}
      real_t   value = 0;

      for(int i=0;i<3;i++){
	double Iminus, Iplus;
	Xfrac[0][i] = 1-modf(X[i]-0.5, &Iminus); // 1-{X[i]-1/2}, floor(X[i]-1/2)
	Xfrac[1][i] = modf(X[i]+0.5,   &Iplus); // {X[i]+1/2}, floor(X[i]+1/2)

	Xint[0][i] = Iminus;
	Xint[1][i] = Iplus;	
      }
    
      // Resample voxel in 2x2x2 neighbourhood
      //000 --- 
      //001 --+
      //010 -+-
      //011 -++
      //100 +--
      //101 +-+
      //110 ++-
      //111 +++

      for(int ijk=0; ijk<=7; ijk++) {
	real_t  weight = 1;
	uint64_t IJK[3];
	
	for(int axis=0;axis<3;axis++){ // x-1/2 or x+1/2
	  int pm = ijk&(1<<axis);
	  IJK[axis] = Xint[pm][axis];
	  weight   *= Xfrac[pm][axis];
	}
	
	voxel_type voxel = voxels.data[IJK[0]+IJK[1]*Nx+IJK[2]*Nx*Ny];
	value += voxel*weight;
      }
      
      plane_samples.data[ui*nv + vj] = value;
    }
}



#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


namespace python_api { 
  namespace py = pybind11;

  typedef py::array_t<voxel_type, py::array::c_style | py::array::forcecast> np_voxelarray;
  typedef py::array_t<real_t, py::array::c_style | py::array::forcecast> np_realarray;  

  array<real_t,3> center_of_mass(const np_voxelarray &np_voxels){
    auto voxels_info    = np_voxels.request();

    return ::center_of_mass({voxels_info.ptr,voxels_info.shape});
  }


  array<real_t,9> inertia_matrix(const np_voxelarray &np_voxels, array<real_t,3>& cm){
    auto voxels_info    = np_voxels.request();
    
    return ::inertia_matrix({voxels_info.ptr,voxels_info.shape}, cm);
  }

  array<real_t,9> inertia_matrix_serial(const np_voxelarray &np_voxels, array<real_t,3>& cm){
    auto voxels_info    = np_voxels.request();
    
    return ::inertia_matrix_serial({voxels_info.ptr,voxels_info.shape}, cm);
  }  


  void sample_plane(const np_voxelarray &np_voxels,
		    const array<real_t,3> &cm,
		    const array<real_t,3> &u_axis,
		    const array<real_t,3> &v_axis,		    
		    np_voxelarray &np_plane_samples,
		    const array<real_t,3> &L)
  {
    auto voxels_info = np_voxels.request();
    auto plane_samples_info  = np_plane_samples.request();
    
    ::sample_plane({voxels_info.ptr, voxels_info.shape},
		   {cm,u_axis,v_axis},
		   {plane_samples_info.ptr, plane_samples_info.shape},
		   L);
  }


  void integrate_axes(const np_voxelarray &np_voxels,
		    const array<real_t,3> &x0,		    
		    const array<real_t,3> &v_axis,
		    const array<real_t,3> &w_axis,
		    const real_t v_min, const real_t w_min,
		    np_realarray &output)
  {
    auto voxels_info = np_voxels.request();
    auto output_info  = output.request();

    ::integrate_axes({voxels_info.ptr, voxels_info.shape},
		     x0,v_axis,w_axis,
		     v_min, w_min,
		     {output_info.ptr, output_info.shape});
  }
}



PYBIND11_MODULE(geometry, m) {
    m.doc() = "Voxel Geometry Module"; // optional module docstring

    m.def("center_of_mass",  &python_api::center_of_mass);
    m.def("inertia_matrix",  &python_api::inertia_matrix);
    m.def("inertia_matrix_serial",  &python_api::inertia_matrix_serial);
    m.def("integrate_axes",    &python_api::integrate_axes);        
    m.def("sample_plane",    &python_api::sample_plane);    
}