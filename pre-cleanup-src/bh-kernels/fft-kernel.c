#include <stdio.h>
#include <stdint.h>
#include <complex.h>

typedef float real_t;
typedef float complex complex_t;

uint32_t reverse_bits_int32(uint32_t ix, int n)
{
    ix = (ix&0xAAAAAAAA)>>1 | (ix&0x55555555)<<1;
    ix = (ix&0xCCCCCCCC)>>2 | (ix&0x33333333)<<2;
    ix = (ix&0xF0F0F0F0)>>4 | (ix&0x0F0F0F0F)<<4;
    ix = (ix&0xFF00FF00)>>8 | (ix&0x00FF00FF)<<8;
    ix = (ix&0xFFFF0000)>>16| (ix&0x0000FFFF)<<16;
    return ix>>(32-n);
}

void bit_reverse_axis0(complex_t *O, uint64_t Om, const complex_t *restrict A, uint64_t Am, uint64_t An, uint64_t n)
{
  /* bit-reverse A: Am x An into O: Om x An */
#pragma omp parallel for
  for(uint32_t i=0;i<Am;i++){ // FFT-axis
    uint32_t i_r = reverse_bits_int32(i,n);
#pragma omp parallel for
    for(uint32_t j=0;j<An;j++)
      O[i_r*An+j] = A[i*An+j];
  }  
}

void bit_reverse_axisn(complex_t *O, uint64_t On, const complex_t *restrict A, uint64_t Am, uint64_t An, uint64_t n)
{
  /* bit-reverse A: Am x An into O: Am x On */
#pragma omp parallel for
    for(uint32_t j=0;j<An;j++)
#pragma omp parallel for
      for(uint32_t i=0;i<Am;i++){ // FFT-axis
	uint32_t i_r = reverse_bits_int32(i,n);
	O[i_r+j*Am] = A[i+j*Am];
  }  
}

void bit_reverse_axisi(complex_t *O, uint64_t On, const complex_t *restrict A, uint64_t Am, uint64_t An, uint64_t n)
{
  /* bit-reverse A: Am x I x An into O: Am x M x On */
#pragma omp parallel for
    for(uint32_t j=0;j<Am;j++)
#pragma omp parallel for
      for(uint32_t i=0;i<Am;i++){ // FFT-axis
	uint32_t i_r = reverse_bits_int32(i,n);
	O[i_r+j*Am] = A[i+j*Am];
  }  
}


/* Radix-2 FFT over axis 0 -- to get a non-power of two FFT, use the Chirp z-transform. */
void fft_axis0(complex_t *restrict O, uint64_t Om, const complex_t *restrict A, const complex_t *restrict w, uint64_t Am, uint64_t An, uint64_t n) 
{
/* Zero-out O */
#pragma omp parallel for
  for(uint64_t i=0;i<Om*An;i++) O[i] = 0;     

  bit_reverse_axis0(O,Om,
		    A,Am,An,
		    n);
  /* FFT-transform */

  for(uint32_t s=1;s<=n;s++){
    printf("Step s=%%d\n",s);
    uint32_t m      = 1<<s;
    uint32_t m_half = m>>1;
    uint32_t w_step = Om/m;	/* 1<<(m-s)? */

    //#pragma omp parallel for	    
    for(uint32_t k=0;k<Om;k+=m)
      //#pragma omp parallel for		      
      for(uint32_t l=0;l<m_half;l++){
	uint64_t i_u = (k+l)*An;
	uint64_t i_t = (k+l+m_half)*An;
	complex_t w_l = w[l*w_step];

#pragma omp parallel for		
	for(uint64_t j=0;j<An;j++){
	  complex_t u = O[i_u+j];
	  complex_t t = O[i_t+j]*w_l;

	  O[i_u+j] = u+t;
	  O[i_t+j] = u-t;
	}
      }
  }
}

/* Radix-2 FFT over any axis: A: Am x M x Ap, M = 2^n
void fft(complex_t *restrict O, uint64_t Om, const complex_t *restrict A, const complex_t *restrict w, uint64_t Am, uint64_t An, uint64_t n) 
{
/* Zero-out O */
#pragma omp parallel for
  for(uint64_t i=0;i<Om*An;i++) O[i] = 0;     

  bit_reverse_axis0(O,Om,
		    A,Am,An,
		    n);
  /* FFT-transform */

  for(uint32_t s=1;s<=n;s++){
    printf("Step s=%%d\n",s);
    uint32_t m      = 1<<s;
    uint32_t m_half = m>>1;
    uint32_t w_step = Om/m;	/* 1<<(m-s)? */

    //#pragma omp parallel for	    
    for(uint32_t k=0;k<Om;k+=m)
      //#pragma omp parallel for		      
      for(uint32_t l=0;l<m_half;l++){
	uint64_t i_u = (k+l)*An;
	uint64_t i_t = (k+l+m_half)*An;
	complex_t w_l = w[l*w_step];

#pragma omp parallel for		
	for(uint64_t j=0;j<An;j++){
	  complex_t u = O[i_u+j];
	  complex_t t = O[i_t+j]*w_l;

	  O[i_u+j] = u+t;
	  O[i_t+j] = u-t;
	}
      }
  }
}


void execute(complex_t *restrict O, const complex_t *restrict A, const complex_t *restrict w){
  uint64_t Am = %(Am)d, An = %(An)d, n = %(n)d;
  uint64_t M  = 1<<n;
  
  fft_axis0(O,M,A,w,Am,An,n);
}
