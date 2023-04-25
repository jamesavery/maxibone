#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Assumes that O.size = sum(image!=0) */
/* On entry:
    O:     empty(sum(image!=0),dtype=uint64) output array
    image: array(image_len)                  image of labels
    num_labels:                              number of segments
 */
typedef %(label_t)s label_t;	/* TODO: parameter */

<<<<<<< Updated upstream
=======
/* Assumes that O.size = sum(image!=0) */
/* On entry:
    O:     empty(sum(image!=0),dtype=uint64) output array
    image: array(image_len)                  image of labels
    num_labels:                              number of segments
 */
typedef uint32_t label_t;	/* TODO: parameter */

>>>>>>> Stashed changes
void collect_labeled(uint64_t *restrict O,
		     const uint64_t *restrict segment_starts, 
		     const label_t  *restrict image,
		     const uint64_t image_len, /* = image.size */
		     const uint64_t num_labels)
{
  uint64_t *offsets = calloc(num_labels,sizeof(uint64_t));

  if(!offsets){
    perror("collect_labeled: error allocating offsets.");
    abort();
  }
  
  for(uint64_t i=0;i<image_len;i++){
    const label_t l = image[i];
    if(l != 0)
      O[segment_starts[l-1] + offsets[l-1]++] = i;
  }

  free(offsets);
}

void execute(uint64_t *restrict O,
	     const uint64_t *restrict segment_starts, 
	     const label_t *restrict image)
{
  collect_labeled(O,segment_starts,image,
		  %(image_len)d,
		  %(num_labels)d);
}
