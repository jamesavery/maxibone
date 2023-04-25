#include <stdio.h>
#include <stdint.h>
#include <complex.h>

typedef %(complex_t)s complex_t;
typedef %(real_t)s    real_t;

/* TODO: Extend to 64 bit? */
uint32_t reverse_bits_int32(uint32_t ix, int n)
{
    ix = (ix&0xAAAAAAAA)>>1 | (ix&0x55555555)<<1;
    ix = (ix&0xCCCCCCCC)>>2 | (ix&0x33333333)<<2;
    ix = (ix&0xF0F0F0F0)>>4 | (ix&0x0F0F0F0F)<<4;
    ix = (ix&0xFF00FF00)>>8 | (ix&0x00FF00FF)<<8;
    ix = (ix&0xFFFF0000)>>16| (ix&0x0000FFFF)<<16;
    return ix>>(32-n);
}


void rfft(complex_t *restrict O,
	  const complex_t *restrict w,
	  const real_t *restrict A, uint64_t Am, uint64_t Ai, uint64_t An,
	  uint64_t n)
{
  const uint64_t M = 1<<n;

  fprintf(stderr,"Zeroing out O.\n");
  
  /* Zero out O */
#pragma omp parallel for
  for(uint64_t i=0;i<Am*M*An;i++) O[i] = 0;       

  fprintf(stderr,"Step 1: zip even and odd coefficients into complex series while reversing bits along transformation axis.\n");
  /* Step 1: zip even and odd coefficients into complex series while reversing bits along transformation axis */
#pragma omp parallel for collapse(3)
  for(uint64_t i=0;i<Am;i++)
    for(uint32_t j=0;j<Ai/2;j++) 
      for(uint64_t k=0;k<An;k++){
	uint32_t I_j = reverse_bits_int32(j,n);
	uint64_t r_idx = 2*j*An+i*Ai*An;
	uint64_t w_idx = I_j*An+i*M*An;
	
	O[w_idx+k] = A[r_idx+k] + I*A[r_idx+An+k];
    }  

  /* Step 2: Usual complex FFT-transform */
  for(uint32_t s=1;s<=n;s++){
    fprintf(stderr,"Step s=%%d\n",s);
    const uint32_t m      = 1<<s;
    const uint32_t m_half = m>>1;
    const uint32_t w_step = M/m;	/* 1<<(m-s)? */

#pragma omp parallel for collapse(4)
    for(uint64_t i=0;i<Am;i++){    
      for(uint32_t k=0;k<M;k+=m){
	for(uint32_t l=0;l<m_half;l++){
	  for(uint64_t j=0;j<An;j++){    	
	    uint64_t  i_u = (k+l)*An        + i*M*An;
	    uint64_t  i_t = (k+l+m_half)*An + i*M*An;
	    complex_t w_l = w[l*w_step];
	    
	    complex_t u = O[i_u+j];
	    complex_t t = O[i_t+j]*w_l;
	    
	    O[i_u+j] = u+t;
	    O[i_t+j] = u-t;
	  }
	}
      }
    }
  }

  fprintf(stderr,"Step 3: Extract X1 and X2 and calculate X1 + w_{2n}^k X2.\n");
  /* Step 3: Extract X1 and X2 and calculate X1 + w_{2n}^k X2*/
  /*     X[1:] = 0.5 *         (X[1:]+np.conj(X[:0:-1])) \ */
  /*           - 0.5j*w2n[1:N]*(X[1:]-np.conj(X[:0:-1])); */
  /*     X[0]  = X[0].real + X[0].imag  */  
  {
    complex_t X1[M], X2[M];

#pragma omp parallel for private(X1,X2) collapse(2)    
    for(uint64_t i=0;i<Am;i++)
      for(uint64_t k=0;k<An;k++){
	for(uint64_t j=1;j<M;j++){
	  uint64_t J  = i*M*An +     j*An+k;
	  uint64_t Jc = i*M*An + (M-j)*An+k;	  
	  X1[j] =  O[J] + conj(O[Jc]);
	  X2[j] =  O[J] - conj(O[Jc]);
	}
	
	for(uint64_t j=1;j<M;j++){
	  uint64_t  J  = i*M*An + j*An+k;
	  O[J] = 0.5*(X1[j] - w[j]*I*X2[j]);
	}
      }

#pragma omp parallel for collapse(2)
    for(uint64_t i=0;i<Am;i++)
      for(uint64_t k=0;k<An;k++){
	uint64_t J0 = i*M*An+k;
	O[J0] = creal(O[J0]) + cimag(O[J0]);
      }
  }

}


void execute(complex_t *restrict O, const complex_t *restrict w, const real_t *restrict A){
  uint64_t Am = %(Am)d, Ai = %(Ai)d, An = %(An)d, n = %(n)d;

  rfft(O,
      w,
      A, Am, Ai, An,
      n);
}
