#include <stdint.h>
#include <stdio.h>

inline doublexs sum(const double *A, int n) {
  double sum = 0;
  for(size_t i=0;i<n;i++) { sum += A[i]; }
  return sum;
}

void sliding_window_1D(const double *A, double *O)
{
  const size_t m = %(m)d, M = %(M)d, w = %(w)d;

#pragma omp parallel for  
  for(size_t i=0;i<M;i++){
    for(size_t j=0; j<m-w; j++){
      size_t A_ix = i*m+j;
      size_t O_ix = i*(m-w)+j;
      O[O_ix] = sum(A+A_ix,w);
    }
  }
}
void execute(const double *A, double *O)
{
  sliding_window_1D(A, O);
}
