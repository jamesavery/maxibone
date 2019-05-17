#include <stdint.h>
#include <stdio.h>
#include <math.h>

#define float_epsilon 1e-%(float_digits)d
typedef %(real_t)s real_t;
typedef long double internal_real_t; // Maximum machine precision 

real_t Froebenius_norm(const real_t *restrict M)
{
  real_t sum = 0;
  for(int i=0;i<9;i++) sum += M[i]*M[i];
  return sqrt(sum);
}


/* Input:
   M: 3 x 3 and symmetric
   Output:
   lambdas: the 3 eigenvalues from smallest to largest absolute value
*/
#define M(i,j) M_data[i*3+j]
#define PI 3.14159265358979323846264338327950288L /* M_PIl is not C99 */

int debug = 0;

void eigenvalues(real_t       *restrict lambdas,
		 const real_t *restrict M_data) 
{
  // Distinct coefficients up to symmetry
  internal_real_t
    a = M(0,0),
    b = M(0,1),
    c = M(0,2),
    d = M(1,1),
    e = M(1,2),
    f = M(2,2);

  // Matrix norm, in order to estimate whether D is "numerically zero".
  real_t M_norm = Froebenius_norm(M_data);
  
  // Coefficients of characteristic polynomial, calculated with Mathematica
  internal_real_t
    A = -1.L,
    B = a+d+f,
    C = b*b + c*c - a*d + e*e - a*f - d*f,
    D = -c*c*d + 2*b*c*e - a*e*e - b*b*f + a*d*f;

  if(debug)
    fprintf(stderr,"(A,B,C,D) = (%%Lf,%%Lf,%%Lf,%%Lf)\n",A,B,C,D);
  
  if(fabs(D/M_norm) < float_epsilon){  	// Characteristic polynomial is second order.
    internal_real_t Disc = sqrtl(B*B-4*A*C);

    lambdas[0] = 0;
    lambdas[1] = (-B-Disc)/(2.L*A);
    lambdas[2] = (-B+Disc)/(2.L*A);
  } else {
    // Depress characteristic polynomial
    // - see http://en.wikipedia.org/wiki/Cubic_equation#Reduction_to_a_depressed_cubic
    internal_real_t 
      p  = (3.L*A*C - B*B)/(3.L*A*A),
      q  = (2.L*B*B*B - 9.L*A*B*C + 27.L*A*A*D)/(27.L*A*A*A),
      xc = B/(3.L*A);

    if(debug)
      fprintf(stderr,"(p,q,xc) = (%%Lf,%%Lf,%%Lf)\n",p,q,xc);

    // François Viète's solution to cubic polynomials with three real roots.
    internal_real_t  K = 2*sqrtl(-p/3.L), arg = (3.L*q)/(2.L*p)*sqrtl(-3.L/p);
    arg = copysign(fminl(1L,fabsl(arg)), arg);

    internal_real_t  theta0 = (1.L/3.L)*acosl(arg);
    if(debug)
      fprintf(stderr,"(K,theta0,arg(theta0)) = (%%Lf,%%Lf,%%Lf)\n",K,3*theta0,arg);
    
    for(int k=0;k<3;k++)
      lambdas[k] = K*cosl(theta0-k*2.L*PI/3.L) - xc;
  }

  if(debug)
    fprintf(stderr,"lambdas = (%%f,%%f,%%f)\n",lambdas[0],lambdas[1],lambdas[2]);
  
  // Sort eigenvalues |l0| >= |l1| >= |l2|
#define swap(i,j) t = lambdas[i]; lambdas[i] = lambdas[j]; lambdas[j] = t;
#define   gt(i,j) fabs(lambdas[i]) > fabs(lambdas[j])
  real_t t;

  if(gt(1,0)){ swap(0,1); }// l0 >= l1
  if(gt(2,0)){ swap(0,2); }// l0 >= l1,l2
  if(gt(2,1)){ swap(1,2); }// l0 >= l1 >= l2


  
  debug = 0;
}

void execute(real_t       *restrict lambdas,
	     const real_t *restrict M_data)
{
#pragma omp parallel for
  for(uint64_t i=0;i < %(num_matrices)d; i++){
    //    if(i==8278) debug = 1;
    eigenvalues(&lambdas[i*3], &M_data[i*9]);
  }
}
