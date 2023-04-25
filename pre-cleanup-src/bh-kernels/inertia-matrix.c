#include <stdint.h>
#include <stdio.h>

void inertia_matrices(float *restrict I,
		      const float *restrict CM,	     /* num_segments x 3 */
		      const uint32_t *restrict segment_xyz, /* num_points x 3 */
		      const uint64_t *restrict segment_starts,
		      const uint64_t num_segments,
		      const uint64_t num_points
		     )
{   
    for(uint64_t l=0;l<num_segments;l++){
      uint64_t n_voxels = segment_starts[l+1] - segment_starts[l]; /* Number of voxels with label l */
      uint64_t l_write = l*9;
      const float *restrict cm = &CM[l*3];
      
      for(uint64_t k=segment_starts[l];k<segment_starts[l+1];k++){
	const uint32_t *restrict x = &segment_xyz[k*3];

	/* Calculate x dot x */
	double xx = 0;
	for(int i=0;i<3;i++) xx += (x[i]-cm[i])*(x[i]-cm[i]);
		
	for(int i=0;i<3;i++){
	  I[l_write+i*3+i] += xx;
	  
	  for(int j=0;j<3;j++)
	    I[l_write+i*3+j] -= (x[i]-cm[i])*(x[j]-cm[j]);
	}
      }
    }
}

void execute(float *restrict I,
	     const float    *restrict CM,
	     const uint32_t *restrict segment_xyz,	     
	     const uint64_t *restrict segment_starts)
{
  inertia_matrices(I,CM,
		   segment_xyz,segment_starts,
		   %(num_segments)d,
		   %(num_points)d);
}
