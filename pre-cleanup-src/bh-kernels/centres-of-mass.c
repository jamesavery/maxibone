#include <stdint.h>
#include <stdio.h>

/* Output:
   cm: num_labels x 3

   Input:
   segment_xyz: num_points x 3
   segment_starts:   num_labels+1
*/
void centres_of_mass(float *restrict CM,
		     const uint32_t *restrict segment_xyz,
		     const uint64_t *restrict segment_starts,
		     const uint64_t num_labels)
{
#pragma omp parallel for
  for(uint64_t i=0;i<num_labels*3;i++) CM[i] = 0;

#pragma omp parallel for collapse(2)
  for(uint64_t l=0;l<num_labels;l++)
    for(int j=0;j<3;j++) {
      uint64_t n_voxels = segment_starts[l+1] - segment_starts[l]; /* Number of voxels with label l */
      float weight = 1.0 / n_voxels;

      for(uint64_t i=0;i<n_voxels;i++)
	CM[l*3+j] += segment_xyz[segment_starts[l]*3+i*3 + j];

      CM[l*3+j] *= weight;
    }
}

void execute(float *restrict CM,
	     const uint32_t *restrict segment_xyz,
	     const uint64_t *restrict segment_starts)
{
  centres_of_mass(CM,segment_xyz,segment_starts,
		  %(num_labels)d);
}
