#include <stdint.h>
#include <stdio.h>

/* Assumes that counts.size = num_labels+1 */
void count_labeled(uint64_t *restrict counts,
		   const uint32_t *labeled_image,
		   const uint64_t image_len, /* = product(labeled_image.shape) */
		   const uint64_t num_labels)
{
  for(uint64_t i=0;i<num_labels+1;i++) counts[i] = 0;

  for(uint64_t i=0,j=0;i<image_len;i++){
    counts[labeled_image[i]]++;
  }
}


void execute(uint64_t *restrict O,
	     const uint32_t *restrict labeled_image)
{
  count_labeled(O,labeled_image,
		%(image_len)d,
		%(num_labels)d
		);
}
