#ifndef boilerplate_h
#define boilerplate_h

// Gaze upon the glory of 3-layered macros for building string literals for _Pragma
#define STRINGIFY(X) #X
#define TOKEN_COMBINER(X) STRINGIFY(X)
#define PRAGMA(X) _Pragma(TOKEN_COMBINER(X))

#ifdef _OPENACC
#define PARALLEL_TERM acc parallel loop
#else
#ifdef _OPENMP
#define PARALLEL_TERM omp parallel for
#else
#define PARALLEL_TERM
#endif
#endif

#ifdef _OPENACC
#define ATOMIC() PRAGMA(acc atomic)
#else
#define ATOMIC() PRAGMA(omp atomic)
#endif

// TODO attempt at docstring; not quite working.

/// Inserts boilerplate code for accessing the given parameter, ARR, in a blocked (chunked) manner.
/// Following this call, the following variables will be exposed:
///
///  - `block_start`: the address of the current block.
///
/// @param ARR The array that will be accessed.
#define FOR_BLOCK_BEGIN(ARR) \
    for (int64_t ARR##_buffer_start = 0; ARR##_buffer_start < ARR##_length; ARR##_buffer_start += acc_block_size<ARR##_type>) { \
        ARR##_type *ARR##_buffer = (ARR##_type *) ARR.data + ARR##_buffer_start; \
        ssize_t ARR##_buffer_length = min(acc_block_size<ARR##_type>, ARR##_length-ARR##_buffer_start); \
        PRAGMA(acc data copy(ARR##_buffer[:ARR##_buffer_length])) \
        {

#define FOR_BLOCK_END() } }

#define FOR_3D_BEGIN(ARR, EXTRA_PRAGMA_CLAUSE) \
    PRAGMA(PARALLEL_TERM collapse(3) EXTRA_PRAGMA_CLAUSE) \
    for (int64_t z = 0; z < ARR##_Nz; z++) { \
        for (int64_t y = 0; y < ARR##_Ny; y++) { \
            for (int64_t x = 0; x < ARR##_Nx; x++) { \

#define FOR_3D_END() }}}

#define FOR_FLAT_BEGIN(ARR, global_prefix, EXTRA_PRAGMA_CLAUSE) \
    PRAGMA(PARALLEL_TERM EXTRA_PRAGMA_CLAUSE) \
    for (int64_t flat_index = 0; flat_index < ARR##_length; flat_index++) { \
        int64_t \
            global_prefix##_index = ARR##_start + flat_index, \
            z = global_prefix##_index / (ARR##_Ny * ARR##_Nx), \
            y = (global_prefix##_index / ARR##_Nx) % ARR##_Ny, \
            x = global_prefix##_index % ARR##_Nx;

#define FOR_FLAT_END() }

#define PUSH_N_DOWN_TO_BUFFER(ARR) \
    ssize_t \
        __attribute__((unused)) ARR##_buffer_Nz = ARR##_Nz, \
        __attribute__((unused)) ARR##_buffer_Ny = ARR##_Ny, \
        __attribute__((unused)) ARR##_buffer_Nx = ARR##_Nx;

#ifdef _OPENACC
#define BLOCK_BEGIN(ARR, EXTRA_PRAGMA_CLAUSE) \
    FOR_BLOCK_BEGIN(ARR) \
    PUSH_N_DOWN_TO_BUFFER(ARR) \
    FOR_FLAT_BEGIN(ARR##_buffer, global, EXTRA_PRAGMA_CLAUSE)

#define BLOCK_END() \
    FOR_FLAT_END() \
    FOR_BLOCK_END()
#else
#ifdef _OPENMP // Should also capture OpenACC, which is why it's second.
#define BLOCK_BEGIN(ARR, EXTRA_PRAGMA_CLAUSE) \
    ARR##_type *ARR##_buffer = (ARR##_type *) ARR.data; \
    FOR_3D_BEGIN(ARR, EXTRA_PRAGMA_CLAUSE) \
    int64_t flat_index = z*ARR##_Ny*ARR##_Nx + y*ARR##_Nx + x;

#define BLOCK_END() FOR_3D_END()
#else
#define BLOCK_BEGIN(ARR, EXTRA_PRAGMA_CLAUSE) \
    int64_t flat_index = 0; \
    ARR##_type *ARR##_buffer = (ARR##_type *) ARR.data; \
    FOR_3D_BEGIN(ARR, EXTRA_PRAGMA_CLAUSE)

#define BLOCK_END() \
    flat_index++; \
    FOR_3D_END()

#endif
#endif

#define UNPACK_NUMPY(ARR) \
    ssize_t \
        __attribute__((unused)) ARR##_Nz = ARR.shape[0], \
        __attribute__((unused)) ARR##_Ny = ARR.shape[1], \
        __attribute__((unused)) ARR##_Nx = ARR.shape[2], \
        __attribute__((unused)) ARR##_length = ARR##_Nz*ARR##_Ny*ARR##_Nx;

#endif