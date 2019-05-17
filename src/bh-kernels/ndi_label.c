#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

typedef %(label_t)s label_t;
typedef %(index_t)s index_t;

#define ndim %(ndim)d


inline bool is_isolated(const label_t *e_neighbours)
{
  bool has_neighbour = false;
  for(int i=0;i<ndim*2;i++) has_neighbour |= e_neighbours[i];
  return !has_neighbour;
}

/* minimum, but 0 counts as infinity */
inline uint64_t minnz(const uint64_t a, const uint64_t b)
{
  return (a-1) < (b-1)? a : b;
}

// TODO: Split and merge to avoid instantiating large images
// On entry:
// L: zeros(shape,dtype=label_t)
// T: arange(len(nonzeros)+1)
// nonzeros, n_nonzeros: sparse representation of bitmap
// On return:
// L: Labeled image
// T: Sparse labels
void ndi_label(label_t *restrict L, label_t *restrict T,
	       const uint64_t total_size,  const int64_t *restrict strides, 
	       const index_t *restrict nonzeros,  const uint64_t n_nonzeros)
{
#pragma omp parallel for
  for(uint64_t i=0;i<n_nonzeros;i++)
    L[nonzeros[i]] = i+1;
  
  label_t num_labels = 0;

  /* Selkow's algorithm generalized to n-D arrays */
  for(uint64_t i=0;i<n_nonzeros;i++){
    int64_t index = nonzeros[i];
    label_t label = L[nonzeros[i]];
    
    label_t e_neighbours[ndim];
    for(int s=0;s<ndim;s++){
      int64_t index_above = index-strides[s];
      e_neighbours[s] = index_above>=0? L[index_above] : 0;
    }    

    label_t a_neighbours[ndim];
    for(int s=0;s<ndim;s++) a_neighbours[s] = T[T[e_neighbours[s]]];
      
    uint64_t e_x = label;
    for(int s=0;s<ndim;s++){
      e_x = minnz(e_x,a_neighbours[s]);
    }
      
    for(int s=0;s<ndim;s++){
      label_t a_k = a_neighbours[s];
      if(a_k != 0 && a_k != e_x){
	T[a_k] = minnz(e_x,T[a_k]);
      }
    }
    T[label] = minnz(e_x,T[label]);
  }

  // Collapse equivalence-relation tree and paint image with labels
  for(uint64_t i=0;i<n_nonzeros;i++){
    T[i+1] = T[T[i+1]];
    L[nonzeros[i]] = T[i+1];
  }
}

void execute(label_t *restrict L, label_t *restrict T,
	     const int64_t *restrict strides,
	     const index_t  *restrict nonzeros)
{
  ndi_label(L,T,
	    %(total_size)d,
	    strides, nonzeros,
	    %(n_nonzeros)d);
}
