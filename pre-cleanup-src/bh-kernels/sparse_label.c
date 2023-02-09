#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

typedef %(label_t)s label_t;
typedef %(index_t)s index_t;

#define ndim        %(ndim)d

/* minimum, but 0 counts as infinity */
inline uint64_t minnz(const uint64_t a, const uint64_t b)
{
  return (a-1) < (b-1)? a : b;
}

uint64_t product(const uint64_t *restrict x, uint64_t n)
{
  uint64_t p = 1;
  for(uint64_t i=0; i<n; i++) p *= x[i];
  return p;
}

void sparse_label(label_t *restrict T,
		  const int64_t *restrict shape,
		  const index_t *restrict nonzeros, const uint64_t n_nonzeros)
{
  /************************************************************************/  
  /*	  INITIALIZATION: For OpenCL kernel, move to Python               */
  /************************************************************************/  
  // Initialize strides (in reverse order for ease of access)
  uint64_t total_size = product(shape,  ndim);
  uint64_t slice_size = product(shape+1,ndim-1);

  // Initialize temporary bitmap-slices L_above and L_active  
  label_t *L_above  = calloc(slice_size,sizeof(label_t));
  label_t *L_active = calloc(slice_size,sizeof(label_t));

  if(L_above == 0 || L_active == 0){
    perror("sparse_label user kernel allocation error:");
    abort();
  }

  // Initialize equivalence tree T
  for(label_t i=0;i<n_nonzeros+1;i++) T[i] = i;

  /**************************************************************************/
  /* ACTUAL CALCULATION: Reconstruct bitmap one slice at a time and connect */
  /* equivalence classes as we sweep down.                                  */
  /**************************************************************************/  
  for(uint64_t z=0,offset=0;z<shape[0];z++){
    // Paint in the bitmap in the current slice
    for(uint64_t i=offset; nonzeros[i] < (z+1)*slice_size && i<n_nonzeros; i++){
      uint64_t full_index  = nonzeros[i];
      uint64_t slice_index = full_index %% slice_size;

      L_active[slice_index] = T[i+1];
    }

    // Now connect each nonzero voxel in the active slice
    for(uint64_t i=offset; nonzeros[i] < (z+1)*slice_size && i<n_nonzeros; i++, offset++){
      int64_t full_index  = nonzeros[i];
      int64_t slice_index = full_index %% slice_size;
      label_t label       = L_active[slice_index];
	
      // Collect predecessor neighbour from each axis
      label_t e_neighbours[ndim];
      e_neighbours[0] = L_above[slice_index];

      for(uint64_t s=0, stride=1;s<ndim-1;s++, stride *= shape[ndim-s-1]){
	int64_t pred_index  = slice_index - stride;
	e_neighbours[s+1]   = pred_index>=0? L_active[pred_index] : 0;
      }    

      // Connect to equivalence class of neighbour with smallest parent
      label_t a_neighbours[ndim];
      for(int s=0;s<ndim;s++) a_neighbours[s] = T[T[e_neighbours[s]]];

      uint64_t e_x = label;
      for(int s=0;s<ndim;s++) e_x = minnz(e_x,a_neighbours[s]);

      for(int s=0;s<ndim;s++){
	label_t a_k = a_neighbours[s];
	if(a_k != 0 && a_k != e_x){
	  T[a_k] = minnz(e_x,T[a_k]);
	}
      }
      T[label] = minnz(e_x,T[label]);
    }

    // Now swap L_active and L_above, and process next slice
    label_t *t = L_active; L_active = L_above; L_above = t;
    for(uint64_t i=0;i<slice_size;i++) L_active[i] = 0;
  }
  free(L_above);
  free(L_active);

  /* COLLAPSING THE EQUIVALENCE TREE yields a sparse representation of the labeled image */
  for(uint64_t i=0;i<n_nonzeros;i++)
    T[i+1] = T[T[i+1]];
}


void execute(label_t *restrict T,
	     const int64_t *restrict shape,
	     const index_t *restrict nonzeros, const uint64_t *restrict n_nonzeros)
{
  sparse_label(T,
	       shape,
	       nonzeros, *n_nonzeros);
}
