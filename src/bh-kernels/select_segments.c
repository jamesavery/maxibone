#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <complex.h>

/* 
   Pick out only those segments s for which selected_segments[s] = True and return new sparse representation
 */

void select_segments(uint64_t *restrict new_indices, uint64_t *restrict new_segment_starts,
		     const uint64_t *restrict segment_indices,	const uint64_t *restrict segment_starts,
		     const uint64_t num_segments,
		     const bool   *restrict selected_segments)
{
  uint64_t l_new = 0, j_new = 0;
  
  for(uint64_t l=0;l<num_segments;l++)
    if(selected_segments[l]){
      new_segment_starts[l_new++] = j_new;

      for(uint64_t j=segment_starts[l]; j<segment_starts[l+1]; j++, j_new++)
	new_indices[j_new] = segment_indices[j];
    }

  new_segment_starts[l_new] = j_new; /* segments end */
}

void execute(      uint64_t *restrict new_indices,            uint64_t *restrict new_segment_starts,
	     const uint64_t *restrict segment_indices,	const uint64_t *restrict segment_starts,
	     const bool  *restrict selected_segments
	     )
{
  select_segments(new_indices, new_segment_starts,
		  segment_indices, segment_starts,
		  %(num_segments)d,
		  selected_segments);
}

