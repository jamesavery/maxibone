#pragma once

#define STR(s) #s
#ifdef _OPENACC
# define reduction_loop(REDUCTION,COPY) _Pragma(STR(acc parallel loop reduction REDUCTION copy COPY))
# define parallel_loop(COPY)            _Pragma(STR(acc parallel loop copy COPY))
# define atomic_statement()             _Pragma("acc atomic")
#else
# ifdef _OPENMP
#  define reduction_loop(REDUCTION,COPY) _Pragma(STR(omp parallel for reduction REDUCTION ))
#  define parallel_loop(COPY)            _Pragma(STR(omp parallel for ))
#  define atomic_statement()             _Pragma("omp atomic update")
# else
#  define reduction_loop(REDUCTION,COPY) 
#  define parallel_loop(COPY)            
#  define atomic_statement()             
# endif
#endif
