#include <stdint.h>
#include <stdio.h>
#include <complex.h>

typedef %(input_t)s  input_t;
typedef %(output_t)s output_t;

/* For example: accumulate(Oilk,Aijk) = "Oilk += Aijk" */
#define accumulate(Oilk,Aijk) %(accumulate)s;

void reduceat(output_t *O,
	      const input_t  *restrict A, const uint64_t Am, const uint64_t Ai, const uint64_t An, 
	      const uint64_t *restrict segment_starts,  const uint64_t num_segments)
{
  const uint64_t
    i_read_stride  = Ai*An,
    i_write_stride = num_segments*An,
    j_stride = An,
    l_stride = An;

#pragma omp parallel for collapse(2)      
  for(uint64_t i=0;i<Am;i++)
    for(uint64_t k=0;k<An;k++)
	for(uint64_t l=0;l<num_segments;l++){

	  uint64_t n_elements    = segment_starts[l+1] - segment_starts[l];
	  uint64_t segment_start = segment_starts[l];

      for(uint64_t j=0;j<n_elements;j++)      
	accumulate(O[i*i_write_stride + l*l_stride + k],
		   A[i*i_read_stride + (segment_start+j)*j_stride + k]);
	}
}

void execute(output_t *O,
	      const input_t  *restrict A, 
	      const uint64_t *restrict segment_starts)
{
  reduceat(O,
	   A, %(Am)d, %(Ai)d, %(An)d, 
	   segment_starts, %(num_segments)d);
}
