inline double sum(global double *A, int n) {
  double sum = 0;
  for(size_t i=0;i<n;i++) { sum += A[i]; }
  return sum;
}

kernel void execute(global double *A, global double *O)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  
  size_t m = %(m)d;
  size_t M = %(M)d;
  size_t w = %(w)d;  
  
  size_t A_ix = i*m+j;
  size_t O_ix = i*(m-w)+j;

  O[O_ix] = sum(A+A_ix,w);
}
