#include <stdio.h>
#include <stdint.h>
#include <complex.h>

typedef %(complex_t)s complex_t;

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
/* A: Am x Ai x An -> O: Am x M x An */
void bit_reverse(complex_t *restrict O, 
		 const complex_t *restrict A, uint64_t Am, uint64_t Ai, uint64_t An, 
		 uint64_t n)
{
  uint64_t M = 1<<n;
  
#pragma omp parallel for collapse(3)
  for(uint64_t i=0;i<Am;i++)
    for(uint32_t j=0;j<Ai;j++) /* TODO: Extend to allow FFT of more than 4G elements */
      for(uint64_t k=0;k<An;k++){
	uint32_t I_j = reverse_bits_int32(j,n);
	O[k+I_j*An+i*M*An] = A[k+j*An+i*Ai*An];
    }
}


/* Radix-2 FFT over axis 0 -- to get a non-power of two FFT, use the Chirp z-transform. */
/* A: Am x Ai x An -> O: Am x (1<<n) x An */
void fft(complex_t *restrict O,
	 const complex_t *restrict w,
	 const complex_t *restrict A, uint64_t Am, uint64_t Ai, uint64_t An,
	 uint64_t n) 
{
  uint64_t M = 1<<n;
/* Zero-out O */
#pragma omp parallel for
  for(uint64_t i=0;i<Am*M*An;i++) O[i] = 0;     

  bit_reverse(O,		/* Bit-reverse along transformation axis */
	      A,Am,Ai,An,
	      n);
  
  /* FFT-transform */
  for(uint32_t s=1;s<=n;s++){
    printf("Step s=%%d\n",s);
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
  printf("FFT is done");
}


   


void execute(complex_t *restrict O, const complex_t *restrict w, const complex_t *restrict A){
  uint64_t Am = %(Am)d, Ai = %(Ai)d, An = %(An)d, n = %(n)d;

  fft(O,
      w,
      A, Am, Ai, An,
      n);
}
