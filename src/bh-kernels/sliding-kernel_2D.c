
void sliding_window_2D(const double *A, double *O/* , uint64_t *shape */)
{
  /* const size_t shape[]  = {%(shape)s}; */
  /* const size_t ndim = %(ndim)d; */
  /* const size_t w1  = %(w1)d; */
  /* const size_t w2  = %(w2)d; */
  const size_t shape[]  = {%(shape)s};
  const size_t ndim = %(ndim)d;
  const size_t w1  = %(w1)d;
  const size_t w2  = %(w2)d;
  

  size_t m1 = shape[0];		// I foerste omgang er axes=0,1.
  size_t m2 = shape[1];
  size_t M = 1;
  for(size_t i=2;i<ndim;i++) M *= shape[i];

 
  size_t A_ix = 0, O_ix = 0;
  for(size_t j=0;j<=M;j++, A_ix+=w1){
    for(size_t i1=0; i1<m1-w1; i1++, A_ix += w2){
      for(size_t i2=0; i2<m2-w2; i2++, A_ix++, O_ix++){
	//	%(fun)s(A+A_ix, O+O_ix);
	O[O_ix] = %(fun)s(A+A_ix,w);
      }
    }
  }
}

