#include <stdint.h>
#include <stdio.h>

/* Output:
   cm: 3 x num_labels 

   Input:
   label_xyz: 3 x num_points 
   row_starts:   num_labels+1
*/
void centres_of_mass(float *restrict cm,
		     const uint32_t *restrict label_xyz,
		     const uint64_t *restrict row_starts,
		     const uint64_t num_labels,
		     const uint64_t num_points
		     )
{
  for(int axis=0;axis<3;axis++){
    uint64_t read_offset  = axis * num_points;
    uint64_t write_offset = axis * num_labels;    
    
    for(uint64_t l=0;l<num_labels;l++){
      uint64_t n_voxels = row_starts[l+1] - row_starts[l]; /* Number of voxels with label l */
      double weight     = 1.0 / n_voxels;
      
      for(uint64_t i=row_starts[l];i<row_starts[l+1];i++)
	cm[write_offset + l] += label_xyz[read_offset + i] * weight;
    }
  }
}

void execute(float *restrict O,
	     const uint32_t *restrict label_xyz,
	     const uint64_t *restrict row_starts)
{
  centres_of_mass(O,label_xyz,row_starts,
		  %(num_labels)d,
		  %(num_points)s);
}
