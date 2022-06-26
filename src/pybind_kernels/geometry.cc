// TODO: Coordinates are named X,Y,Z in c++, but Z,Y,X in python. Homogenize to X,Y,Z!
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


// TODO: Fix OpenACC copies & re-enable GPU
array<real_t,3> center_of_mass(const input_ndarray<mask_type> voxels) {
  // nvc++ doesn't support OpenACC 2.7 array reductions yet.  
  real_t cmx = 0, cmy = 0, cmz = 0;
  size_t  Nx = voxels.shape[0], Ny = voxels.shape[1], Nz = voxels.shape[2];
  int64_t image_length = Nx*Ny*Nz;

  print_timestamp("center_of_mass start");
  real_t total_mass = 0;  
  for(int64_t block_start=0;block_start<image_length;block_start+=acc_block_size){

    const mask_type *buffer = voxels.data + block_start;
    ssize_t this_block_length = min(acc_block_size,image_length-block_start);

    //#pragma acc parallel loop reduction(+:cmx,cmy,cmz,total_mass) copyin(buffer[:this_block_length])
    reduction_loop((+:cmx,cmy,cmz,total_mass),())
    for(int64_t k = 0; k<this_block_length;k++){
      real_t          m = buffer[k];      

      int64_t flat_idx = block_start + k;
      int64_t x = flat_idx / (Ny*Nz);
      int64_t y = (flat_idx / Nz) % Ny;
      int64_t z = flat_idx % Nz;

      total_mass += m;
      cmx += m*x; cmy += m*y; cmz += m*z;
    }
  }
  cmx /= total_mass; cmy /= total_mass; cmz /= total_mass;
  
  print_timestamp("center_of_mass end");  

  return array<real_t,3>{cmx,cmy,cmz};
}


array<real_t,9> inertia_matrix_serial(const input_ndarray<mask_type> &voxels, const array<real_t,3> &cm)
{
  real_t
    Ixx = 0, Ixy = 0, Ixz = 0,
             Iyy = 0, Iyz = 0,
                      Izz = 0;
  
  ssize_t Nx = voxels.shape[0], Ny = voxels.shape[1], Nz = voxels.shape[2];

  print_timestamp("inertia_matrix_serial start");
  for(int64_t X=0,k=0;X<Nx;X++)
    for(int64_t Y=0;Y<Ny;Y++)
      for(int64_t Z=0;Z<Nz;Z++,k++){
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


array<real_t,9> inertia_matrix(const input_ndarray<mask_type> &voxels, const array<real_t,3> &cm)
{
  // nvc++ doesn't support OpenACC 2.7 array reductions yet, so must name each element.
  real_t
    M00 = 0, M01 = 0, M02 = 0,
             M11 = 0, M12 = 0,
                      M22 = 0;
  
  size_t Nx = voxels.shape[0], Ny = voxels.shape[1], Nz = voxels.shape[2];
  ssize_t image_length = Nx*Ny*Nz;

  print_timestamp("inertia_matrix start");    
  for(ssize_t block_start=0;block_start<image_length;block_start+=acc_block_size){
    const mask_type *buffer  = voxels.data + block_start;
    ssize_t block_length = min(acc_block_size,image_length-block_start);

    reduction_loop((+:M00,M01,M02,M11,M12,M22),())
    for(int64_t k = 0; k<block_length;k++) {    //\if(buffer[k] != 0)
	int64_t flat_idx = block_start + k;
	real_t xs[3] = {(flat_idx  / (Ny*Nz))  - cm[0],  // x
			((flat_idx / Nz) % Ny) - cm[1],  // y
			(flat_idx  % Nz)       - cm[2]}; // z

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


void integrate_axes(const input_ndarray<mask_type> &voxels,
		    const array<real_t,3> &x0,		    
		    const array<real_t,3> &v_axis,
		    const array<real_t,3> &w_axis,
		    const real_t v_min, const real_t w_min,
		    output_ndarray<real_t> output)
{
  ssize_t Nx = voxels.shape[0], Ny = voxels.shape[1], Nz = voxels.shape[2];
  ssize_t Nv = output.shape[0], Nw = output.shape[1]; 
  int64_t image_length = Nx*Ny*Nz;
  real_t *output_data = output.data;

  // TODO: Check v_axis & w_axis projections to certify bounds and get rid of runtime check
  
  for(ssize_t block_start=0;block_start<image_length;block_start += acc_block_size){
    const mask_type *buffer  = voxels.data + block_start;
    int block_length = min(acc_block_size,image_length-block_start);

    //#pragma acc parallel loop copy(output_data[:Nv*Nw]) copyin(buffer[:block_length], x0, v_axis, w_axis)
    parallel_loop((output_data[:Nv*Nw]))
    for(int64_t k = 0; k<block_length;k++) if(buffer[k] != 0) {
	int64_t flat_idx = block_start + k;
	real_t xs[3] = {(flat_idx  / (Ny*Nz))  - x0[0],  // x
			((flat_idx / Nz) % Ny) - x0[1],  // y
			(flat_idx  % Nz)       - x0[2]}; // z

	mask_type voxel = buffer[k];
	real_t v = dot(xs,v_axis), w = dot(xs,w_axis);
	int64_t i_v = round(v-v_min), j_w = round(w-w_min);

	if(i_v >= 0 && j_w >= 0 && i_v < Nv && j_w < Nw){
	  atomic_statement()
	  output_data[i_v*Nw + j_w] += voxel;
	}
      }
  }
}


bool in_bbox(float U, float V, float W, const std::array<float,6> bbox)
{
  const auto& [U_min,U_max,V_min,V_max,W_min,W_max] = bbox;

  bool inside = U>=U_min && U<=U_max && V>=V_min && V<=V_max && W>=W_min && W<=W_max;

  // printf("in_bbox: (%.1f,%.1f,%.1f) \in ([%.1f,%.1f],[%.1f,%.1f],[%.1f,%.1f]) == %d\n",
  // 	 U,V,W,U_min,U_max,V_min,V_max,U_min,U_max,inside);
  return inside;
}


template<typename field_type> float resample2x2x2(const field_type      *voxels,
						  const array<ssize_t,3> &shape,
						  const array<float,3>   &X)
{
  auto  [Nx,Ny,Nz] = shape;	// Eller omvendt?
  if(!in_bbox(X[0],X[1],X[2], {0.5,Nx-1.5, 0.5,Ny-1.5, 0.5,Nz-1.5})){
    uint64_t voxel_index = X[0]*Ny*Nz+X[1]*Ny+X[2];      
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
    // if(I<0 || J<0 || K<0){
    //   printf("(I,J,K) = (%ld,%ld,%ld)\n",I,J,K);
    //   abort();
    // }
    // if(I>=int(Nx) || J>=int(Ny) || K>=int(Nz)){
    //   printf("(I,J,K) = (%ld,%ld,%ld), (Nx,Ny,Nz) = (%ld,%ld,%ld)\n",I,J,K,Nx,Ny,Nz);
    //   abort();
    // }
    uint64_t voxel_index = I*Ny*Nz+J*Ny+K;
    field_type voxel = voxels[voxel_index];
    value += voxel*weight;
  }
  return value;
}

template <typename voxel_type> void sample_plane(const input_ndarray<voxel_type> &voxels,
						 const real_t voxel_size, // In micrometers
						 const array<real_t,3> cm,
						 const array<real_t,3> u_axis,
						 const array<real_t,3> v_axis,		  
						 const array<real_t,4>  bbox,    // [umin,umax,vmin,vmax] in micrometers
						 output_ndarray<real_t> plane_samples)
{
  const auto& [umin,umax,vmin,vmax] = bbox; // In micrometers
  ssize_t Nx = voxels.shape[0], Ny = voxels.shape[1], Nz = voxels.shape[2];
  ssize_t nu = plane_samples.shape[0], nv = plane_samples.shape[1];
  real_t  du = (umax-umin)/nu, dv = (vmax-vmin)/nv;

  #pragma omp parallel for collapse(2)
  for(ssize_t ui=0;ui<nu;ui++)
    for(ssize_t vj=0;vj<nv;vj++){
      const real_t u = umin + ui*du, v = vmin + vj*dv;

      // X,Y,Z in micrometers;  x,y,z in voxel index space      
      const real_t		
	X = cm[0] + u*u_axis[0] + v*v_axis[0],
	Y = cm[1] + u*u_axis[1] + v*v_axis[1],
	Z = cm[2] + u*u_axis[2] + v*v_axis[2];

      const real_t x = X/voxel_size, y = Y/voxel_size, z = Z/voxel_size;

      //      printf("u,v = %g,%g -> %.1f,%.1f,%.1f -> %d, %d, %d\n",u,v,X,Y,Z,int(round(x)),int(round(y)),int(round(z)));
      
      voxel_type value = 0;
      if(in_bbox(x,y,z,{0.5,Nx-0.5, 0.5,Ny-0.5, 0.5,Nz-0.5}))
	value = resample2x2x2<voxel_type>(voxels.data,{Nx,Ny,Nz},{x,y,z});
      // else
      // 	fprintf(stderr,"Sampling outside image: x,y,z = %.1f,%.1f,%.1f, Nx,Ny,Nz = %ld,%ld,%ld\n",x,y,z,Nx,Ny,Nz);

      plane_samples.data[ui*nv + vj] = value;
    }
}

// NB: xyz are in indices, not micrometers
void zero_outside_bbox(const array<real_t,9> &principal_axes,
		       const array<real_t,6> &parameter_ranges,
		       const array<real_t,3> &cm,
		       output_ndarray<mask_type> voxels)
{
  size_t  Nx = voxels.shape[0], Ny = voxels.shape[1], Nz = voxels.shape[2];
  int64_t image_length = Nx*Ny*Nz;

  printf("(Nx,Ny,Nz) = (%ld,%ld,%ld), image_length = %ld",Nx,Ny,Nz, image_length);
  for(int64_t block_start=0;block_start<image_length;block_start+=acc_block_size){

    mask_type *buffer = voxels.data + block_start;
    ssize_t this_block_length = min(acc_block_size,image_length-block_start);

    parallel_loop((buffer[:this_block_length]))
    for(int64_t k = 0; k<this_block_length;k++){
      int64_t flat_idx = block_start + k;
      int64_t x = flat_idx  / (Ny*Nz);
      int64_t y = (flat_idx / Nz) % Ny;
      int64_t z = flat_idx  % Nz;
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

inline vector4 hom_transform(const vector4 &x, const matrix4x4 &M)
{
  vector4 c{{0,0,0,0}};

  for(int i=0;i<4;i++){
    real_t sum = 0;
#pragma simd parallel for reduction(+:sum)    
    for(int j=0;j<4;j++)
      sum += M[i*4+j]*x[j];
    c[i] = sum;
  }
  return c;

  
}



#define loop_mask_start(mask_in,mask_out,COPY) {		               \
  ssize_t Mx = mask_in.shape[0], My = mask_in.shape[1], Mz = mask_in.shape[2]; \
  ssize_t mask_length = Mx*My*Mz;                                              \
                                                                               \
for(ssize_t block_start=0;block_start<mask_length;block_start+=acc_block_size){\
  const mask_type *maskin_buffer  = mask_in.data + block_start;                \
        mask_type *maskout_buffer = mask_out.data + block_start;               \
  ssize_t  this_block_length = min(acc_block_size,mask_length-block_start);    \
  									\
  _Pragma(STR(acc parallel loop copy(maskin_buffer[:this_block_length], maskout_buffer[:this_block_length]) copy COPY)) \
  for(int64_t k = 0; k<this_block_length;k++){                                 \
    int64_t flat_idx = block_start + k;                                        \
    int64_t X = (flat_idx  / (My*Mz)), Y = (flat_idx / Mz) % My, Z = flat_idx  % Mz; \
    std::array<real_t,4> Xs = {X*voxel_size, Y*voxel_size, Z*voxel_size, 1}; \
    bool mask_value = maskin_buffer[k];

#define loop_mask_end(mask) }}} 


void fill_implant_mask(const input_ndarray<mask_type> implant_mask,
		       float voxel_size,
		       const array<float,6> &bbox,
		       float r_fraction,
		       const matrix4x4 &Muvw,
		       output_ndarray<mask_type> solid_implant_mask,
		       output_ndarray<float> rsqr_maxs,
		       output_ndarray<float> profile
		       )
{
  real_t theta_min = M_PI, theta_max = -M_PI;  
  ssize_t n_segments = rsqr_maxs.shape[0];
  const auto [U_min,U_max,V_min,V_max,W_min,W_max] = bbox;
  
  printf("implant_mask.shape = %ld,%ld,%ld\n",implant_mask.shape[0],implant_mask.shape[1],implant_mask.shape[2]);
  printf("solid_implant_mask.shape = %ld,%ld,%ld\n",solid_implant_mask.shape[0],solid_implant_mask.shape[1],solid_implant_mask.shape[2]);
  
  fprintf(stderr,"voxel_size = %g, U_min = %g, U_max = %g, r_frac = %g, n_segments = %ld\n",
	 voxel_size, U_min, U_max, r_fraction, n_segments);

  float     *rsqr_maxs_d     = rsqr_maxs.data;
  float     *profile_d       = profile.data;
  
  // First pass computes some bounds -- possibly separate out to avoid repeating
  loop_mask_start(implant_mask, solid_implant_mask,
		  (maskin_buffer[:this_block_length], rsqr_maxs_d[:n_segments], Muvw[:16], bbox[:6]) );
  if(mask_value){
    auto [U,V,W,c] = hom_transform(Xs,Muvw);
    
    real_t r_sqr = V*V+W*W;
    real_t theta = atan2(V,W);

    int U_i = floor((U-U_min)*(n_segments-1)/(U_max-U_min));

    //    if(U_i >= 0 && U_i < n_segments){
    if( in_bbox(U,V,W,bbox) ){
      rsqr_maxs_d[U_i] = max(rsqr_maxs_d[U_i], float(r_sqr));
      theta_min = min(theta_min, theta);
      theta_max = max(theta_max, theta);
      //      W_min     = min(W_min,     W);
    } else {
      // Otherwise we've calculated it wrong!
      //  fprintf(stderr,"U-coordinate out of bounds: U_i = %ld, U = %g, U_min = %g, U_max = %g\n",U_i,U,U_min,U_max);
    }
  }
  loop_mask_end(implant_mask);

  double theta_center = (theta_max+theta_min)/2;

  fprintf(stderr,"theta_min, theta_center, theta_max = %g,%g,%g\n", theta_min, theta_center, theta_max);

  // Second pass does the actual work
  loop_mask_start(implant_mask, solid_implant_mask,
		  (rsqr_maxs_d[:n_segments], profile_d[:n_segments]) );
  auto [U,V,W,c] = hom_transform(Xs,Muvw);
  float r_sqr = V*V+W*W;
  float theta = atan2(V,W);
  int U_i = floor((U-U_min)*(n_segments-1)/(U_max-U_min));

  bool solid_mask_value = false;
  if(U_i >= 0 && U_i < n_segments && W>=W_min){ // TODO: Full bounding box check?
    solid_mask_value = mask_value | (r_sqr <= r_fraction*rsqr_maxs_d[U_i]);

    if(theta >= theta_min && theta <= theta_center && r_sqr <= rsqr_maxs_d[U_i]){
      atomic_statement()
	profile_d[U_i] += solid_mask_value;
    }
  }
  maskout_buffer[k] = solid_mask_value;
  
  loop_mask_end(implant_mask);
}
		       
void compute_front_mask(const input_ndarray<mask_type> solid_implant,
		const float voxel_size,
		const matrix4x4 &Muvw,		
		std::array<float,6> bbox,
		output_ndarray<mask_type> front_mask)
{
  const auto [U_min,U_max,V_min,V_max,W_min,W_max] = bbox;

  loop_mask_start(solid_implant, front_mask,
		  () );  

  if(!mask_value){
    auto [U,V,W,c] = hom_transform(Xs,Muvw);

    maskout_buffer[k] = W>W_min;
  } else
    maskout_buffer[k] = 0;
  
  loop_mask_end(solid_implant)
}


void cylinder_projection(const input_ndarray<float>  edt,  // Euclidean Distance Transform in um, should be low-resolution (will be interpolated)
			 const input_ndarray<uint8_t> Cs,  // Material classification images (probability per voxel, 0..1 -> 0..255)
			 float voxel_size,		   // Voxel size for Cs
			 float d_min, float d_max,	   // Distance shell to map to cylinder
			 float theta_min, float theta_max, // Angle range (wrt cylinder center)
			 std::array<float,6> bbox,
			 const matrix4x4 &Muvw,		   // Transform from zyx (in um) to U'V'W' cylinder FoR (in um)
			 output_ndarray<float>    images,  // Probability-weighted volume of (class,theta,U)-voxels
			 output_ndarray<int64_t>  counts   // Number of (class,theta,U)-voxels
			 )
{
  ssize_t n_images = images.shape[0], n_theta = images.shape[1], n_U = images.shape[2];

  const auto& [U_min,U_max,V_min,V_max,W_min,W_max] = bbox;

  ssize_t                     ex = edt.shape[0], ey = edt.shape[1], ez = edt.shape[2];
  ssize_t  nC = Cs.shape[0],  Cx = Cs.shape[1],  Cy = Cs.shape[2],  Cz = Cs.shape[3];

  real_t edx = ex/real_t(Cx), edy = ey/real_t(Cy), edz = ex/real_t(Cz);
  
  assert(nC == n_images);  

  ssize_t edt_length       = ex*ey*ez;
  ssize_t C_length         = Cx*Cy*Cz;  
  ssize_t C_strides[4]     = {Cx*Cy*Cz,Cy*Cz,Cz,1};
  ssize_t image_strides[3] = {n_theta*n_U,n_U,1};  

  printf("Segmenting from %g to %g micrometers distance of implant.\n",d_min,d_max);

  printf("Bounding box is [U_min,U_max,V_min,V_max,W_min,W_max] = [[%g,%g],[%g,%g],[%g,%g]]\n",
	 U_min,U_max,V_min,V_max,W_min,W_max);
  printf("EDT field is (%ld,%ld,%ld)\n",ex,ey,ez);
  
  real_t th_min = 1234, th_max = -1234;
  ssize_t n_shell = 0;
  ssize_t n_shell_bbox = 0;

  ssize_t block_height = 128;
  
  //TODO: new acc/openmp macro in parallel.hh
  {    
    for(int c=0;c<nC;c++){   // TODO: Skal denne være udenfor?
      const uint8_t *C = &Cs.data[c*C_strides[0]];
      auto *image = &images.data[c*image_strides[0]];
      auto *count = &counts.data[c*image_strides[0]];

      for(ssize_t block_start=0, edt_block_start=0;
	  block_start<C_length;
	  block_start+=block_height*Cy*Cz, edt_block_start+=block_height*ey*ez){
	const uint8_t *C_buffer = C + block_start;
	const float  *edt_block = edt.data + max(block_start-ey*ez,0L);

	ssize_t  this_block_length = min(block_height*Cy*Cz,C_length-block_start);
	ssize_t  this_edt_length   = min((block_height+2)*ey*ez,edt_length-block_start);

#pragma acc parallel loop copy(C_buffer[:this_block_length], image[:n_theta*n_U], count[:n_theta*n_U], bbox[:6], Muvw[:16], edt_block[:this_edt_length]) reduction(+:n_shell,n_shell_bbox)
	//#pragma omp parallel for reduction(+:n_shell,n_shell_bbox)	
	for(int64_t k = 0; k<this_block_length;k++){	
	  const int64_t flat_idx = block_start + k;
	  const int64_t X = (flat_idx  / (Cy*Cz)), Y = (flat_idx / Cz) % Cy, Z = flat_idx  % Cz; // Integer indices: Cs[c,X,Y,Z]
	  // Index into local block
	  const int64_t Xl = (k  / (Cy*Cz)), Yl = (k / Cz) % Cy, Zl = k  % Cz;
	  // Index into local edt block. Note EDT has 1-slice padding top+bottom
	  const float  x = (Xl+1)*edx, y = Yl*edy, z = Zl*edy;
	  
	  //****** MEAT OF THE IMPLEMENTATION IS HERE ******
	  real_t distance = resample2x2x2<float>(edt_block,{this_edt_length/(ey*ez),ey,ez},
						 {x,y,z});
	  
	  if(distance > d_min && distance <= d_max){ // TODO: and W>w_min
	    array<real_t,4> Xs = {X*voxel_size, Y*voxel_size, Z*voxel_size, 1};
	    auto [U,V,W,c] = hom_transform(Xs,Muvw);
	    n_shell ++;

	    //	    printf("distance = %.1f, U,V,W = %.2f,%.2f,%.2f\n",distance,U,V,W);
	    if(in_bbox(U,V,W,bbox)){
	      n_shell_bbox++;
	      
	      real_t theta    = atan2(V,W);
	      
	      size_t theta_i = floor( (theta-theta_min) * (n_theta-1)/(theta_max-theta_min) );
	      size_t U_i     = floor( (U    -    U_min) * (n_U    -1)/(    U_max-    U_min) );
	      
	      real_t p = C_buffer[k]/255.;

	      if(p>0){
		th_min = min(theta,th_min);
		th_max = max(theta,th_max);	      
		
		atomic_statement()
		  image[theta_i*n_U + U_i] += p;
	      
		atomic_statement()	  
		  count[theta_i*n_U + U_i] += 1;
	      }
	    }
	  }
	}
      }
    }
  }
  printf("n_shell = %ld, n_shell_bbox = %ld\n",n_shell,n_shell_bbox);
  printf("theta_min, theta_max = %.2f,%.2f\n",theta_min,theta_max);
  printf("th_min,       th_max = %.2f,%.2f\n",th_min,th_max);    
}

