#include <stdint.h>
#include <stdio.h>
#include <math.h>

typedef %(real_t)s real_t;
typedef long double internal_real_t;

void eigenvectors(real_t *restrict x,
		  const real_t lambda,		  
		  const real_t *restrict M_data)
{
  // [ a_12 * a_23 - a_13 * (a_22 - r) ]
  // [ a_12 * a_13 - a_23 * (a_11 - r) ]
  // [(a_11 - r) * (a_22 - r) - a_12^2 ]
  // [ be - c*(d - r) ]
  // [ bc - e*(a - r) ]
  // [ (a-r)*(d-r) - b^2 ]  
#define M(i,j) M_data[i*3+j]
  internal_real_t a = M(0,0), b = M(0,1), c = M(0,2),
                              d = M(1,1), e = M(1,2),
                                          f = M(2,2);
  
  x[0] = b*e - (d-lambda)*c;
  x[1] = (a-lambda)*(f-lambda)-c*c;  
  x[1] = b*c - e*(a-lambda);


  real_t norm = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
  if(norm == 0)			/* Redo permuted? */
    ;
  
  for(int i=0;i<3;i++) x[i] *= 1/norm; 
}


void execute(      real_t *restrict xs,
	     const real_t *restrict lambdas,
	     const real_t *restrict M_data)
{
#pragma omp parallel for collapse(2)
  for(uint64_t i=0;i < %(num_matrices)d; i++)
    for(int j=0;j<3;j++)
    eigenvectors(&xs[i*9+j*3], lambdas[i*3+j], &M_data[i*9]);
}
