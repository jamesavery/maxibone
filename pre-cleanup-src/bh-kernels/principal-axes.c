#include <stdint.h>
#include <stdio.h>
#include <math.h>

/* TODO: general types. */

eigenvector(float *restrict x,
	    const float *restrict M_data,
	    const float lambda)
{
  // [ a_12 * a_23 - a_13 * (a_22 - r) ]
  // [ a_12 * a_13 - a_23 * (a_11 - r) ]
  // [(a_11 - r) * (a_22 - r) - a_12^2 ]
#define M(i,j) = M_data[i*3+j]

  x[0] = M(0,1)*M(1,2) - M(0,2)*(M(1,1)-lambda);
  x[1] = M(0,1)*M(0,2) - M(1,2)*(M(0,0)-lambda);
  x[2] = (M(0,0)-lambda)*(M(1,1)-lambda)-M(0,1)*M(0,1);

  float norm = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);

  for(i=0;i<3;i++) x[i] *= 1/norm; 
}


eigensystem3x3(float *restrict lambdas,
	       float *restrict U,
	       const float *restrict A
	       ) const {
  eigenvalues(lambdas,A);


    // Sort eigenvalues by absolute value, smallest first
    if(fabs(lambda[0]) > fabs(lambda[1])) std::swap(lambda[0],lambda[1]);
    if(fabs(lambda[1]) > fabs(lambda[2])) std::swap(lambda[1],lambda[2]);
    if(fabs(lambda[0]) > fabs(lambda[1])) std::swap(lambda[0],lambda[1]);

    // Build eigenvector matrix
    for(int i=0;i<3;i++){
      coord3d c(eigenvector(lambda[i]));
      for(int j=0;j<3;j++) C(i,j) = c[j];
    }
    return make_pair(lambda,C);
  }


matrix3d Polyhedron::principal_axes() const
{
  const matrix3d I(inertia_matrix());
  pair<coord3d,matrix3d> ES(I.eigensystem());

  matrix3d Id;
  Id(0,0) = 1; 
  Id(1,1) = 1; 
  Id(2,2) = 1; 
/*
  cerr << "Inertial frame:\n " 
       << " inertia_matrix = " << I << ";\n"
       << " lambda  = " << ES.first << ";\n"
       << " vectors = " << ES.second << ";\n";
*/

  for(int i=0;i<3;i++) 
    if(std::isnan(ES.first[i])){
      cerr << "Warning: Inertial frame returned NaN. Setting inertial frame transformation to identity.\n";
      return Id;
    }
  
  if((ES.second*ES.second.transpose() - Id).norm() > 1e-2){
    cerr << "Warning: Inertial frame transform is not unitary. Setting inertial frame transformation to identity.\n";
    return Id;
  }

  return ES.second;
}


void inertia_matrices(float *restrict I,
		      const float *restrict CM,	     /* num_segments x 3 */
		      const uint32_t *restrict segment_xyz, /* num_points x 3 */
		      const uint64_t *restrict segment_starts,
		      const uint64_t num_segments,
		      const uint64_t num_points
		     )
{   
    for(uint64_t l=0;l<num_segments;l++){
      uint64_t n_voxels = segment_starts[l+1] - segment_starts[l]; /* Number of voxels with label l */
      uint64_t l_write = l*9;
      const float *restrict cm = &CM[l*3];
      
      for(uint64_t k=segment_starts[l];k<segment_starts[l+1];k++){
	const uint32_t *restrict x = &segment_xyz[k*3];

	/* Calculate x dot x */
	double xx = 0;
	for(int i=0;i<3;i++) xx += (x[i]-cm[i])*(x[i]-cm[i]);
		
	for(int i=0;i<3;i++){
	  I[l_write+i*3+i] += xx;
	  
	  for(int j=0;j<3;j++)
	    I[l_write+i*3+j] -= (x[i]-cm[i])*(x[j]-cm[j]);
	}
      }
    }
}

void execute(float *restrict I,
	     const float    *restrict CM,
	     const uint32_t *restrict segment_xyz,	     
	     const uint64_t *restrict segment_starts)
{
  inertia_matrices(I,CM,
		   segment_xyz,segment_starts,
		   %(num_segments)d,
		   %(num_points)d);
}
