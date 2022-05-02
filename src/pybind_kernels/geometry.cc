#include <chrono>
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <math.h>
using namespace std;

#include "datatypes.hh"
#include "parallel.hh"

#define dot(a,b) (a[0]*b[0] + a[1]*b[1] + a[2]*b[2])

void print_timestamp(string message)
{
  auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
  tm local_tm = *localtime(&now);
  fprintf(stderr,"%s at %02d:%02d:%02d\n", message.c_str(), local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);    
}


array<real_t,3> center_of_mass(const input_ndarray<voxel_type> voxels) {
  // nvc++ doesn't support OpenACC 2.7 array reductions yet.  
  real_t cmx = 0, cmy = 0, cmz = 0;
  size_t  Nz = voxels.shape[0], Ny = voxels.shape[1], Nx = voxels.shape[2];
  int64_t image_length = Nx*Ny*Nz;

  print_timestamp("center_of_mass start");
  real_t total_mass = 0;  
  for(int64_t block_start=0;block_start<image_length;block_start+=acc_block_size){

    const voxel_type *buffer = voxels.data + block_start;
    ssize_t this_block_length = min(acc_block_size,image_length-block_start);

    //#pragma acc parallel loop reduction(+:cmx,cmy,cmz,total_mass) copyin(buffer[:this_block_length])
    reduction_loop((+:cmx,cmy,cmz,total_mass),())
    for(int64_t k = 0; k<this_block_length;k++){
      real_t          m = buffer[k];      

      int64_t flat_idx = block_start + k;
      int64_t x = flat_idx % Nx;
      int64_t y = (flat_idx / Nx) % Ny;
      int64_t z = flat_idx / (Nx*Ny);

      total_mass += m;
      cmx += m*x; cmy += m*y; cmz += m*z;
    }
  }
  cmx /= total_mass; cmy /= total_mass; cmz /= total_mass;
  
  print_timestamp("center_of_mass end");  

  return array<real_t,3>{cmx,cmy,cmz};
}


array<real_t,9> inertia_matrix_serial(const input_ndarray<voxel_type> &voxels, const array<real_t,3> &cm)
{
  real_t
    Ixx = 0, Ixy = 0, Ixz = 0,
             Iyy = 0, Iyz = 0,
                      Izz = 0;
  
  ssize_t Nz = voxels.shape[0], Ny = voxels.shape[1], Nx = voxels.shape[2];

  print_timestamp("inertia_matrix_serial start");
  for(int64_t Z=0,k=0;Z<Nz;Z++)
    for(int64_t Y=0;Y<Ny;Y++)
      for(int64_t X=0;X<Nx;X++,k++){
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


array<real_t,9> inertia_matrix(const input_ndarray<voxel_type> &voxels, const array<real_t,3> &cm)
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

    reduction_loop((+:M00,M01,M02,M11,M12,M22),())
    for(int64_t k = 0; k<block_length;k++) {    //\if(buffer[k] != 0)
	int64_t flat_idx = block_start + k;
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


void integrate_axes(const input_ndarray<voxel_type> &voxels,
		    const array<real_t,3> &x0,		    
		    const array<real_t,3> &v_axis,
		    const array<real_t,3> &w_axis,
		    const real_t v_min, const real_t w_min,
		    output_ndarray<real_t> output)
{
  ssize_t Nz = voxels.shape[0], Ny = voxels.shape[1], Nx = voxels.shape[2];
  ssize_t Nv = output.shape[0], Nw = output.shape[1]; 
  int64_t image_length = Nx*Ny*Nz;
  real_t *output_data = output.data;

  // TODO: Check v_axis & w_axis projections to certify bounds and get rid of runtime check
  
  for(ssize_t block_start=0;block_start<image_length;block_start += acc_block_size){
    const voxel_type *buffer  = voxels.data + block_start;
    int block_length = min(acc_block_size,image_length-block_start);

    //#pragma acc parallel loop copy(output_data[:Nv*Nw]) copyin(buffer[:block_length], x0, v_axis, w_axis)
    parallel_loop((output_data[:Nv*Nw]))
    for(int64_t k = 0; k<block_length;k++) if(buffer[k] != 0) {
	int64_t flat_idx = block_start + k;
	real_t xs[3] = {(flat_idx % Nx)        - x0[0],  // x
			((flat_idx / Nx) % Ny) - x0[1],  // y
			(flat_idx / (Nx*Ny))   - x0[2]}; // z

	voxel_type voxel = buffer[k];
	real_t v = dot(xs,v_axis), w = dot(xs,w_axis);
	int64_t i_v = round(v-v_min), j_w = round(w-w_min);

	if(i_v >= 0 && j_w >= 0 && i_v < Nv && j_w < Nw){
	  atomic_statement()
	  output_data[i_v*Nw + j_w] += voxel;
	}
      }
  }
}

template <typename t> real_t resample2x2x2(const input_ndarray<t> &voxels,
					   const real_t &x, const real_t &y, const real_t &z)
{
  ssize_t  Nz = voxels.shape[0], Ny = voxels.shape[1], Nx = voxels.shape[2];
  assert(x>=0.5 && y>=0.5 && z>= 0.5);
  assert(x<=Nx-0.5 && y<=Ny-0.5 && z<= Nz-0.5);  
				    
  real_t   X[3] = {x,y,z};
  real_t   Xfrac[2][3];	// {Xminus[3], Xplus[3]}
  int64_t Xint[2][3];	// {Iminus[3], Iplus[3]}
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
    int64_t IJK[3];
	
    for(int axis=0;axis<3;axis++){ // x-1/2 or x+1/2
      int pm = ijk&(1<<axis);
      IJK[axis] = Xint[pm][axis];
      weight   *= Xfrac[pm][axis];
    }
	
    voxel_type voxel = voxels.data[IJK[0]+IJK[1]*Nx+IJK[2]*Nx*Ny];
    value += voxel*weight;
  }
  return value;
}

void sample_plane(const input_ndarray<voxel_type> &voxels, const plane_t &plane, output_ndarray<voxel_type> plane_samples, array<real_t,3> L)
{
  ssize_t Ny = voxels.shape[1], Nx = voxels.shape[2];
  ssize_t nu = plane_samples.shape[0], nv = plane_samples.shape[1];  
  real_t  du = L[0]/nu, dv = L[1]/nv;

  for(ssize_t ui=0;ui<nu;ui++)
    for(ssize_t vj=0;vj<nv;vj++){
      real_t u = (ui-nu/2)*du, v = (vj-nv/2)*dv;

      real_t x = plane.cm[0] + u*plane.u_axis[0] + v*plane.v_axis[0], y = plane.cm[1] + u*plane.u_axis[1] + v*plane.v_axis[1], z = plane.cm[2] + u*plane.u_axis[2] + v*plane.v_axis[2];

      voxel_type value = resample2x2x2(voxels,x,y,z);      
      if(x >= 0.5 && y >= 0.5 && x+0.5 <= Nx && y+0.5 <= Ny){
	plane_samples.data[ui*nv + vj] = value;
      }
    }
}

// NB: xyz are in indices, not micrometers
void zero_outside_bbox(const array<real_t,9> &principal_axes,
		       const array<real_t,6> &parameter_ranges,
		       const array<real_t,3> &cm,
		       output_ndarray<voxel_type> voxels)
{
  size_t  Nz = voxels.shape[0], Ny = voxels.shape[1], Nx = voxels.shape[2];
  int64_t image_length = Nx*Ny*Nz;

  printf("(Nx,Ny,Nz) = (%ld,%ld,%ld), image_length = %ld",Nx,Ny,Nz, image_length);
  for(int64_t block_start=0;block_start<image_length;block_start+=acc_block_size){

    voxel_type *buffer = voxels.data + block_start;
    ssize_t this_block_length = min(acc_block_size,image_length-block_start);

    parallel_loop((buffer[:this_block_length]))
    for(int64_t k = 0; k<this_block_length;k++){
      int64_t flat_idx = block_start + k;
      int64_t x = flat_idx % Nx;
      int64_t y = (flat_idx / Nx) % Ny;
      int64_t z = flat_idx / (Nx*Ny);
      // Boilerplate until here. TODO: macroize or lambda out!
      
      real_t xs[3] = {x-cm[0], y-cm[1], z-cm[2]};

      real_t params[3] = {0,0,0};

      for(int uvw=0;uvw<3;uvw++)
	for(int xyz=0;xyz<3;xyz++)
	  params[uvw] += xs[xyz]*principal_axes[uvw*3+xyz]; // u = dot(xs,u_axis), v = dot(xs,v_axis), w = dot(xs,w_axis)

      bool p = false;

      for(int uvw=0;uvw<3;uvw++){
	real_t param_min = parameter_ranges[uvw*2], param_max = parameter_ranges[uvw*2+1];
	p |= (params[uvw] < param_min) | (params[uvw] > param_max);
      }

      if(p) buffer[k] = 0;

    }
  }
}

void field_histogram_resample(const input_ndarray<uint16_t> voxels,
			      const input_ndarray<uint16_t> field,
			      output_ndarray<uint64_t> &bins,
			      const double vmin, const double vmax)
{
  const ssize_t
    nZ = voxels.shape[0], nY = voxels.shape[1], nX = voxels.shape[2],
    nz = field.shape[0],  ny = field.shape[1],  nx = field.shape[2],
    nbins = bins.shape[0];

  const real_t dx = nx/double(nX), dy = ny/double(nY), dz = nz/double(nZ);

  // TODO: copyin voxels z and field z,z+1
  for(ssize_t Z=0;Z<nZ;Z++)
    for(ssize_t Y=0;Y<nY;Y++)
      for(ssize_t X=0;X<nX;X++){
	real_t x = X*dx, y = Y*dy, z = Z*dz;

	uint16_t  voxel_value = voxels.data[Z*nY*nX+Y*nX+X];
	uint16_t  field_value = round(resample2x2x2(field,x,y,z));

	bins.data[(voxel_value>>4)*nbins + field_value]++;
      }
    
}

