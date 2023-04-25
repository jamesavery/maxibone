#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

void execute(uint64_t *restrict nonzero, const bool *restrict bitmap)
{
  for(uint64_t i=0,j=0;i<%(image_len)d;i++){
    if(bitmap[i]) nonzero[j++] = i;
  }
}
