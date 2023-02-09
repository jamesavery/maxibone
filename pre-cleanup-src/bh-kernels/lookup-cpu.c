#include <stdint.h>
#include <stdio.h>
#include <complex.h>

typedef %(ctype)s scalar_t;


/* A: Am x Ai x An -> Am x Ii x An */
void index_axis(scalar_t *O,
		scalar_t *A, uint64_t Am, uint64_t Ai, uint64_t An,
		int64_t *Index, uint64_t Ii)
{
#pragma omp parallel for collapse(3)
  for(uint64_t i=0;i<Am;i++)
    for(uint64_t j=0;j<Ii;j++)
      for(uint64_t k=0;k<An;k++){
	uint64_t I_j = Index[j]>=0? Index[j] : Ai+Index[j];
	O[k+j*An+i*Ii*An] = A[k+I_j*An+i*Ai*An];
      }
}



void execute(scalar_t *O, scalar_t *A, int64_t *Index)
{
  uint64_t Am = %(Am)d, Ai = %(Ai)d, Ii = %(Ii)d, An =%(An)d;

  return index_axis(O, A,Am,Ai,An, Index,Ii);
}
