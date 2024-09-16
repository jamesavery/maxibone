/**
 * @file boilerplate.hh
 * Boilerplate code for parallelizing array operations.
 */
#ifndef boilerplate_h
#define boilerplate_h

//
// Gaze upon the glory of 3-layered macros for building string literals for _Pragma:
//

/**
 * Stringifies the given argument. This is a helper macro for the `PRAGMA` macro.
 *
 * @param `X` The argument to stringify.
 */
#define STRINGIFY(X) #X
/**
 * Combines the given argument with the `STRINGIFY` macro. This is a helper macro for the `PRAGMA` macro.
 *
 * @param `X` The argument to combine.
 */
#define TOKEN_COMBINER(X) STRINGIFY(X)
/**
 * Generates a pragma string for the given argument. This is used to generate different pragmas based on the compiler. For example, `PRAGMA(acc parallel loop)` will generate `#pragma acc parallel loop` for OpenACC, while `PRAGMA(omp parallel for)` will generate `#pragma omp parallel for` for OpenMP.
 * @param `X` The tokens to combine into a pragma string.
 */
#define PRAGMA(X) _Pragma(TOKEN_COMBINER(X))

#ifdef _OPENACC
/**
 * Inserts boilerplate code for parallelizing a loop. This macro will generate the appropriate pragma for the compiler. For example, `PARALLEL_TERM()` will generate `#pragma acc parallel loop` for OpenACC, while `PARALLEL_TERM()` will generate `#pragma omp parallel for` for OpenMP. If neither OpenACC nor OpenMP is detected, this macro will do nothing.
 */
#define PARALLEL_TERM() PRAGMA(acc parallel loop)
#elif _OPENMP
/**
 * Inserts boilerplate code for parallelizing a loop. This macro will generate the appropriate pragma for the compiler. For example, `PARALLEL_TERM()` will generate `#pragma acc parallel loop` for OpenACC, while `PARALLEL_TERM()` will generate `#pragma omp parallel for` for OpenMP. If neither OpenACC nor OpenMP is detected, this macro will do nothing.
 */
#define PARALLEL_TERM() PRAGMA(omp parallel for)
#else
/**
 * Inserts boilerplate code for parallelizing a loop. This macro will generate the appropriate pragma for the compiler. For example, `PARALLEL_TERM()` will generate `#pragma acc parallel loop` for OpenACC, while `PARALLEL_TERM()` will generate `#pragma omp parallel for` for OpenMP. If neither OpenACC nor OpenMP is detected, this macro will do nothing.
 */
#define PARALLEL_TERM()
#endif

#ifdef _OPENACC
/**
 * Inserts boilerplate code for atomic operations. This macro will generate the appropriate pragma for the compiler. For example, `ATOMIC()` will generate `#pragma acc atomic` for OpenACC, while `ATOMIC()` will generate `#pragma omp atomic` for OpenMP. If neither OpenACC nor OpenMP is detected, this macro will do nothing.
 */
#define ATOMIC() PRAGMA(acc atomic)
#elif _OPENMP
/**
 * Inserts boilerplate code for atomic operations. This macro will generate the appropriate pragma for the compiler. For example, `ATOMIC()` will generate `#pragma acc atomic` for OpenACC, while `ATOMIC()` will generate `#pragma omp atomic` for OpenMP. If neither OpenACC nor OpenMP is detected, this macro will do nothing.
 */
#define ATOMIC() PRAGMA(omp atomic)
#else
/**
 * Inserts boilerplate code for atomic operations. This macro will generate the appropriate pragma for the compiler. For example, `ATOMIC()` will generate `#pragma acc atomic` for OpenACC, while `ATOMIC()` will generate `#pragma omp atomic` for OpenMP. If neither OpenACC nor OpenMP is detected, this macro will do nothing.
 */
#define ATOMIC()
#endif

/**
 * Inserts boilerplate code for accessing the given parameter, `ARR`, in a blocked (chunked) manner.
 *
 * Following this call, the following variables will be exposed:
 *
 * - `ARR_buffer_start`: the address of the current block.
 *
 * - `ARR_buffer`: the pointer to the current block, i.e. `ARR.data + ARR_buffer_start`.
 *
 * - `ARR_buffer_length`: the length of the current block.
 *
 * For OpenACC enabled compilation, `ARR_buffer[:ARR_buffer_length]` will be copied to the device.
 *
 * @param ARR The array that will be accessed.
 */
#define FOR_BLOCK_BEGIN(ARR) \
    for (int64_t ARR##_buffer_start = 0; ARR##_buffer_start < ARR##_length; ARR##_buffer_start += acc_block_size<ARR##_type>) { \
        ARR##_type *ARR##_buffer = (ARR##_type *) ARR.data + ARR##_buffer_start; \
        ssize_t ARR##_buffer_length = min(acc_block_size<ARR##_type>, ARR##_length-ARR##_buffer_start); \
        PRAGMA(acc data copy(ARR##_buffer[:ARR##_buffer_length])) \
        {

/**
 * Closes the block started by `FOR_BLOCK_BEGIN`.
 */
#define FOR_BLOCK_END() } }

/**
 * Inserts boilerplate code for accessing the given parameter, `ARR`, in a blocked (chunked) manner.
 * This version is for templated types, where `ARR` has type `T`.
 *
 * Following this call, the following variables will be exposed:
 *
 * - `ARR_buffer_start`: the address of the current block.
 *
 * - `ARR_buffer`: the pointer to the current block, i.e. `ARR.data + ARR_buffer_start`.
 *
 * - `ARR_buffer_length`: the length of the current block.
 *
 * For OpenACC enabled compilation, `ARR_buffer[:ARR_buffer_length]` will be copied to the device.
 *
 * @param ARR The array that will be accessed.
 */
#define FOR_BLOCK_BEGIN_T(ARR) \
    for (int64_t ARR##_buffer_start = 0; ARR##_buffer_start < ARR##_length; ARR##_buffer_start += acc_block_size<T>) { \
        T *ARR##_buffer = (T *) ARR.data + ARR##_buffer_start; \
        ssize_t ARR##_buffer_length = std::min(acc_block_size<T>, ARR##_length-ARR##_buffer_start); \
        PRAGMA(acc data copy(ARR##_buffer[:ARR##_buffer_length])) \
        {

/**
 * Closes the block started by `FOR_BLOCK_BEGIN_T`.
 */
#define FOR_BLOCK_END_T() } }

/**
 * Inserts boilerplate code for accessing the given parameter, `ARR_IN`, in a blocked (chunked) manner, and also provides an output array, `ARR_OUT`.
 * This macro assumes that the input and output arrays have the same length.
 *
 * Following this call, the following variables will be exposed:
 *
 * - `ARR_IN_buffer_start`: the address of the current block.
 *
 * - `ARR_IN_buffer`: the pointer to the current block, i.e. `ARR_IN.data + ARR_IN_buffer_start`.
 *
 * - `ARR_OUT_buffer`: the pointer to the current block in the output array, i.e. `ARR_OUT.data + ARR_IN_buffer_start`.
 *
 * - `ARR_IN_buffer_length`: the length of the current block.
 *
 * For OpenACC enabled compilation, `ARR_IN_buffer[:ARR_IN_buffer_length]` will be copied to the device, and `ARR_OUT_buffer[:ARR_IN_buffer_length]` will be copied back to the host.
 *
 * @param ARR_IN The input array that will be accessed.
 * @param ARR_OUT The output array that will will be accessed.
 */
#define FOR_BLOCK_BEGIN_WITH_OUTPUT(ARR_IN, ARR_OUT) \
    for (int64_t ARR_IN##_buffer_start = 0; ARR_IN##_buffer_start < ARR_IN##_length; ARR_IN##_buffer_start += acc_block_size<ARR_IN##_type> / 2) { \
        ARR_IN##_type *ARR_IN##_buffer = (ARR_IN##_type *) ARR_IN.data + ARR_IN##_buffer_start; \
        ARR_OUT##_type *ARR_OUT##_buffer = (ARR_OUT##_type *) ARR_OUT.data + ARR_IN##_buffer_start; \
        ssize_t ARR_IN##_buffer_length = min(acc_block_size<ARR_IN##_type>, ARR_IN##_length - ARR_IN##_buffer_start); \
        PRAGMA(acc data copyin(ARR_IN##_buffer[:ARR_IN##_buffer_length]) copy(ARR_OUT##_buffer[:ARR_IN##_buffer_length])) \
        {

/**
 * Closes the block started by `FOR_BLOCK_BEGIN_WITH_OUTPUT`.
 */
#define FOR_BLOCK_END_WITH_OUTPUT() } }

/**
 * Inserts boilerplate code for accessing the given parameter, `ARR_IN`, in a blocked (chunked) manner, and also provides an output array, `ARR_OUT`.
 * This macro assumes that the input and output arrays have the same length.
 * This version is for templated types, where `ARR_IN` has type `T` and `ARR_OUT` has type `U`.
 *
 * Following this call, the following variables will be exposed:
 *
 * - `ARR_IN_buffer_start`: the address of the current block.
 *
 * - `ARR_IN_buffer`: the pointer to the current block, i.e. `ARR_IN.data + ARR_IN_buffer_start`.
 *
 * - `ARR_OUT_buffer`: the pointer to the current block in the output array, i.e. `ARR_OUT.data + ARR_IN_buffer_start`.
 *
 * - `ARR_IN_buffer_length`: the length of the current block.
 *
 * For OpenACC enabled compilation, `ARR_IN_buffer[:ARR_IN_buffer_length]` will be copied to the device, and `ARR_OUT_buffer[:ARR_IN_buffer_length]` will be copied back to the host.
 *
 * @param ARR_IN The input array that will be accessed.
 * @param ARR_OUT The output array that will will be accessed.
 */
#define FOR_BLOCK_BEGIN_WITH_OUTPUT_TU(ARR_IN, ARR_OUT) \
    for (int64_t ARR_IN##_buffer_start = 0; ARR_IN##_buffer_start < ARR_IN##_length; ARR_IN##_buffer_start += acc_block_size<T> / 2) { \
        T *ARR_IN##_buffer = (T *) ARR_IN.data + ARR_IN##_buffer_start; \
        U *ARR_OUT##_buffer = (U *) ARR_OUT.data + ARR_IN##_buffer_start; \
        ssize_t ARR_IN##_buffer_length = std::min(acc_block_size<T>, ARR_IN##_length - ARR_IN##_buffer_start); \
        PRAGMA(acc data copyin(ARR_IN##_buffer[:ARR_IN##_buffer_length]) copy(ARR_OUT##_buffer[:ARR_IN##_buffer_length])) \
        {

/**
 * Closes the block started by `FOR_BLOCK_BEGIN_WITH_OUTPUT_TU`.
 */
#define FOR_BLOCK_END_WITH_OUTPUT_TU() } }

/**
 * Sets up traversal of a 3D array, `ARR` and adds the relevant pragma clauses.
 * It also allows for additional pragma clauses, `EXTRA_PRAGMA_CLAUSE`, to be added to the parallel loop.
 * E.g. `FOR_3D_BEGIN(ARR, reduction(+:sum))` will generate `#pragma acc parallel loop reduction(+:sum)` for OpenACC and `#pragma omp parallel for reduction(+:sum)` for OpenMP.
 *
 * Following this call, the following variables will be exposed:
 *
 * - `z`: the current z-index.
 *
 * - `y`: the current y-index.
 *
 * - `x`: the current x-index.
 *
 * @param ARR The array that will be traversed.
 * @param EXTRA_PRAGMA_CLAUSE Additional pragma clauses to be added to the parallel loop.
 */
#define FOR_3D_BEGIN(ARR, EXTRA_PRAGMA_CLAUSE) \
    PRAGMA(PARALLEL_TERM collapse(3) EXTRA_PRAGMA_CLAUSE) \
    for (int64_t z = 0; z < ARR##_Nz; z++) { \
        for (int64_t y = 0; y < ARR##_Ny; y++) { \
            for (int64_t x = 0; x < ARR##_Nx; x++) { \

/**
 * Closes the block started by `FOR_3D_BEGIN`.
 */
#define FOR_3D_END() }}}

/**
 * Sets up traversal of a 3D array, `ARR` and adds the relevant pragma clauses.
 * This function differs from `FOR_3D_BEGIN` in that it uses one for loop, rather than three nested loops.
 * It will prefix all of the exposed variables with the `GLOBAL_PREFIX` parameter.
 * It also allows for additional pragma clauses, `EXTRA_PRAGMA_CLAUSE`, to be added to the parallel loop.
 * E.g. `FOR_3D_BEGIN_T(ARR, reduction(+:sum))` will generate `#pragma acc parallel loop reduction(+:sum)` for OpenACC and `#pragma omp parallel for reduction(+:sum)` for OpenMP.
 *
 * Following this call, the following variables will be exposed:
 *
 * - `flat_index`: the current flat index.
 *
 * - `GLOBAL_PREFIX_index`: the current global index.
 *
 * - `GLOBAL_PREFIX_z`: the current z-index.
 *
 * - `GLOBAL_PREFIX_y`: the current y-index.
 *
 * - `GLOBAL_PREFIX_x`: the current x-index.
 *
 * @param ARR The array that will be traversed.
 * @param GLOBAL_PREFIX The prefix for the global indices.
 * @param EXTRA_PRAGMA_CLAUSE Additional pragma clauses to be added to the parallel loop.
 */
#define FOR_FLAT_BEGIN(ARR, GLOBAL_PREFIX, EXTRA_PRAGMA_CLAUSE) \
    PRAGMA(PARALLEL_TERM EXTRA_PRAGMA_CLAUSE) \
    for (int64_t flat_index = 0; flat_index < ARR##_length; flat_index++) { \
        int64_t \
            __attribute__((unused)) GLOBAL_PREFIX##_index = ARR##_start + flat_index, \
            __attribute__((unused)) z = GLOBAL_PREFIX##_index / (ARR##_Ny * ARR##_Nx), \
            __attribute__((unused)) y = (GLOBAL_PREFIX##_index / ARR##_Nx) % ARR##_Ny, \
            __attribute__((unused)) x = GLOBAL_PREFIX##_index % ARR##_Nx;

/**
 * Closes the block started by `FOR_FLAT_BEGIN`.
 */
#define FOR_FLAT_END() }

/**
 * Pushes the sizes of the array `ARR` down to the buffers (ARR_buffer) sizes.
 *
 * Following this call, the following variables will be exposed:
 *
 * - `ARR_buffer_Nz`: the size of the array in the z-dimension.
 *
 * - `ARR_buffer_Ny`: the size of the array in the y-dimension.
 *
 * - `ARR_buffer_Nx`: the size of the array in the x-dimension.
 *
 * @param ARR The array whose sizes will be pushed down to the buffer.
 */
#define PUSH_N_DOWN_TO_BUFFER(ARR) \
    ssize_t \
        __attribute__((unused)) ARR##_buffer_Nz = ARR##_Nz, \
        __attribute__((unused)) ARR##_buffer_Ny = ARR##_Ny, \
        __attribute__((unused)) ARR##_buffer_Nx = ARR##_Nx;

#ifdef _OPENACC

/**
 * Inserts boilerplate code for accessing the given parameter, `ARR`, in a blocked (chunked) manner.
 * For OpenACC, this macro will first start a blocked outer loop, push down the sizes and then start a flat inner loop.
 * For OpenMP, this macro will start a 3D loop.
 * This macro allows for additional pragma clauses, `EXTRA_PRAGMA_CLAUSE`, to be added to the parallel loop.
 * E.g. `BLOCK_BEGIN(ARR, reduction(+:sum))` will generate `#pragma acc parallel loop reduction(+:sum)` for OpenACC and `#pragma omp parallel for reduction(+:sum)` for OpenMP.
 *
 * @param ARR The array that will be accessed.
 * @param EXTRA_PRAGMA_CLAUSE Additional pragma clauses to be added to the parallel loop.
 */
#define BLOCK_BEGIN(ARR, EXTRA_PRAGMA_CLAUSE) \
    FOR_BLOCK_BEGIN(ARR) \
    PUSH_N_DOWN_TO_BUFFER(ARR) \
    FOR_FLAT_BEGIN(ARR##_buffer, global, EXTRA_PRAGMA_CLAUSE)

/**
 * Closes the block started by `BLOCK_BEGIN`.
 */
#define BLOCK_END() \
    FOR_FLAT_END() \
    FOR_BLOCK_END()

/**
 * Inserts boilerplate code for accessing the given parameter, `ARR`, in a blocked (chunked) manner, where the array has type `T`.
 * For OpenACC, this macro will first start a blocked outer loop, push down the sizes and then start a flat inner loop.
 * For OpenMP, this macro will start a 3D loop.
 * This macro allows for additional pragma clauses, `EXTRA_PRAGMA_CLAUSE`, to be added to the parallel loop.
 * E.g. `BLOCK_BEGIN_T(ARR, reduction(+:sum))` will generate `#pragma acc parallel loop reduction(+:sum)` for OpenACC and `#pragma omp parallel for reduction(+:sum)` for OpenMP.
 *
 * @param ARR The array that will be accessed.
 * @param EXTRA_PRAGMA_CLAUSE Additional pragma clauses to be added to the parallel loop.
 */
#define BLOCK_BEGIN_T(ARR, EXTRA_PRAGMA_CLAUSE) \
    FOR_BLOCK_BEGIN_T(ARR) \
    PUSH_N_DOWN_TO_BUFFER(ARR) \
    FOR_FLAT_BEGIN(ARR##_buffer, global, EXTRA_PRAGMA_CLAUSE)

/**
 * Closes the block started by `BLOCK_BEGIN_T`.
 */
#define BLOCK_END_T() \
    FOR_FLAT_END() \
    FOR_BLOCK_END_T()

/**
 * Inserts boilerplate code for accessing the given parameter, `ARR_IN`, in a blocked (chunked) manner, and also provides an output array, `ARR_OUT`, also in a blocked manner.
 * This macro assumes that the input and output arrays have the same length.
 * For OpenACC, this macro will first start a blocked outer loop, push down the sizes and then start a flat inner loop.
 * For OpenMP, this macro will start a 3D loop.
 * This macro allows for additional pragma clauses, `EXTRA_PRAGMA_CLAUSE`, to be added to the parallel loop.
 * E.g. `BLOCK_BEGIN_WITH_OUTPUT(ARR_IN, ARR_OUT, reduction(+:sum))` will generate `#pragma acc parallel loop reduction(+:sum)` for OpenACC and `#pragma omp parallel for reduction(+:sum)` for OpenMP.
 *
 * @param ARR_IN The input array that will be accessed.
 * @param ARR_OUT The output array that will will be accessed.
 * @param EXTRA_PRAGMA_CLAUSE Additional pragma clauses to be added to the parallel loop.
 */
#define BLOCK_BEGIN_WITH_OUTPUT(ARR_IN, ARR_OUT, EXTRA_PRAGMA_CLAUSE) \
    FOR_BLOCK_BEGIN_WITH_OUTPUT(ARR_IN, ARR_OUT) \
    PUSH_N_DOWN_TO_BUFFER(ARR_IN) \
    FOR_FLAT_BEGIN(ARR_IN##_buffer, global, EXTRA_PRAGMA_CLAUSE)

/**
 * Closes the block started by `BLOCK_BEGIN_WITH_OUTPUT`.
 */
#define BLOCK_END_WITH_OUTPUT() \
    FOR_FLAT_END() \
    FOR_BLOCK_END_WITH_OUTPUT()

/**
 * Inserts boilerplate code for accessing the given parameter, `ARR_IN`, in a blocked (chunked) manner, and also provides an output array, `ARR_OUT`, also in a blocked manner.
 * This macro assumes that the input and output arrays have the same length.
 * This version is for templated types, where `ARR_IN` has type `T` and `ARR_OUT` has type `U`.
 * For OpenACC, this macro will first start a blocked outer loop, push down the sizes and then start a flat inner loop.
 * For OpenMP, this macro will start a 3D loop.
 * This macro allows for additional pragma clauses, `EXTRA_PRAGMA_CLAUSE`, to be added to the parallel loop.
 * E.g. `BLOCK_BEGIN_WITH_OUTPUT_TU(ARR_IN, ARR_OUT, reduction(+:sum))` will generate `#pragma acc parallel loop reduction(+:sum)` for OpenACC and `#pragma omp parallel for reduction(+:sum)` for OpenMP.
 *
 * @param ARR_IN The input array that will be accessed.
 * @param ARR_OUT The output array that will will be accessed.
 */
#define BLOCK_BEGIN_WITH_OUTPUT_TU(ARR_IN, ARR_OUT, EXTRA_PRAGMA_CLAUSE) \
    FOR_BLOCK_BEGIN_WITH_OUTPUT_TU(ARR_IN, ARR_OUT) \
    PUSH_N_DOWN_TO_BUFFER(ARR_IN) \
    FOR_FLAT_BEGIN(ARR_IN##_buffer, global, EXTRA_PRAGMA_CLAUSE)

/**
 * Closes the block started by `BLOCK_BEGIN_WITH_OUTPUT_TU`.
 */
#define BLOCK_END_WITH_OUTPUT_TU() \
    FOR_FLAT_END() \
    FOR_BLOCK_END_WITH_OUTPUT_TU()

#else
#ifdef _OPENMP // Should also capture OpenACC, which is why it's second.

/**
 * Inserts boilerplate code for accessing the given parameter, `ARR`, in a blocked (chunked) manner.
 * For OpenACC, this macro will first start a blocked outer loop, push down the sizes and then start a flat inner loop.
 * For OpenMP, this macro will start a 3D loop.
 * This macro allows for additional pragma clauses, `EXTRA_PRAGMA_CLAUSE`, to be added to the parallel loop.
 * E.g. `BLOCK_BEGIN(ARR, reduction(+:sum))` will generate `#pragma acc parallel loop reduction(+:sum)` for OpenACC and `#pragma omp parallel for reduction(+:sum)` for OpenMP.
 *
 * @param ARR The array that will be accessed.
 * @param EXTRA_PRAGMA_CLAUSE Additional pragma clauses to be added to the parallel loop.
 */
#define BLOCK_BEGIN(ARR, EXTRA_PRAGMA_CLAUSE) \
    ARR##_type *ARR##_buffer = (ARR##_type *) ARR.data; \
    __attribute__((unused)) int64_t ARR##_buffer_start = 0; \
    FOR_3D_BEGIN(ARR, EXTRA_PRAGMA_CLAUSE) \
    int64_t flat_index = z*ARR##_Ny*ARR##_Nx + y*ARR##_Nx + x;

/**
 * Closes the block started by `BLOCK_BEGIN`.
 */
#define BLOCK_END() FOR_3D_END()

/**
 * Inserts boilerplate code for accessing the given parameter, `ARR`, in a blocked (chunked) manner, where the array has type `T`.
 * For OpenACC, this macro will first start a blocked outer loop, push down the sizes and then start a flat inner loop.
 * For OpenMP, this macro will start a 3D loop.
 * This macro allows for additional pragma clauses, `EXTRA_PRAGMA_CLAUSE`, to be added to the parallel loop.
 * E.g. `BLOCK_BEGIN_T(ARR, reduction(+:sum))` will generate `#pragma acc parallel loop reduction(+:sum)` for OpenACC and `#pragma omp parallel for reduction(+:sum)` for OpenMP.
 *
 * @param ARR The array that will be accessed.
 * @param EXTRA_PRAGMA_CLAUSE Additional pragma clauses to be added to the parallel loop.
 */
#define BLOCK_BEGIN_T(ARR, EXTRA_PRAGMA_CLAUSE) \
    T *ARR##_buffer = (T *) ARR.data; \
    __attribute__((unused)) int64_t ARR##_buffer_start = 0; \
    FOR_3D_BEGIN(ARR, EXTRA_PRAGMA_CLAUSE) \
    int64_t flat_index = z*ARR##_Ny*ARR##_Nx + y*ARR##_Nx + x;

/**
 * Closes the block started by `BLOCK_BEGIN_T`.
 */
#define BLOCK_END_T() FOR_3D_END()

/**
 * Inserts boilerplate code for accessing the given parameter, `ARR_IN`, in a blocked (chunked) manner, and also provides an output array, `ARR_OUT`, also in a blocked manner.
 * This macro assumes that the input and output arrays have the same length.
 * For OpenACC, this macro will first start a blocked outer loop, push down the sizes and then start a flat inner loop.
 * For OpenMP, this macro will start a 3D loop.
 * This macro allows for additional pragma clauses, `EXTRA_PRAGMA_CLAUSE`, to be added to the parallel loop.
 * E.g. `BLOCK_BEGIN_WITH_OUTPUT(ARR_IN, ARR_OUT, reduction(+:sum))` will generate `#pragma acc parallel loop reduction(+:sum)` for OpenACC and `#pragma omp parallel for reduction(+:sum)` for OpenMP.
 *
 * @param ARR_IN The input array that will be accessed.
 * @param ARR_OUT The output array that will will be accessed.
 * @param EXTRA_PRAGMA_CLAUSE Additional pragma clauses to be added to the parallel loop.
 */
#define BLOCK_BEGIN_WITH_OUTPUT(ARR_IN, ARR_OUT, EXTRA_PRAGMA_CLAUSE) \
    ARR_IN##_type *ARR_IN##_buffer = (ARR_IN##_type *) ARR_IN.data; \
    ARR_OUT##_type *ARR_OUT##_buffer = (ARR_OUT##_type *) ARR_OUT.data; \
    __attribute__((unused)) int64_t ARR_IN##_buffer_start = 0; \
    FOR_3D_BEGIN(ARR_IN, EXTRA_PRAGMA_CLAUSE) \
    int64_t flat_index = z*ARR_IN##_Ny*ARR_IN##_Nx + y*ARR_IN##_Nx + x;

/**
 * Closes the block started by `BLOCK_BEGIN_WITH_OUTPUT`.
 */
#define BLOCK_END_WITH_OUTPUT() FOR_3D_END()

/**
 * Inserts boilerplate code for accessing the given parameter, `ARR_IN`, in a blocked (chunked) manner, and also provides an output array, `ARR_OUT`, also in a blocked manner.
 * This macro assumes that the input and output arrays have the same length.
 * This version is for templated types, where `ARR_IN` has type `T` and `ARR_OUT` has type `U`.
 * For OpenACC, this macro will first start a blocked outer loop, push down the sizes and then start a flat inner loop.
 * For OpenMP, this macro will start a 3D loop.
 * This macro allows for additional pragma clauses, `EXTRA_PRAGMA_CLAUSE`, to be added to the parallel loop.
 * E.g. `BLOCK_BEGIN_WITH_OUTPUT_TU(ARR_IN, ARR_OUT, reduction(+:sum))` will generate `#pragma acc parallel loop reduction(+:sum)` for OpenACC and `#pragma omp parallel for reduction(+:sum)` for OpenMP.
 *
 * @param ARR_IN The input array that will be accessed.
 * @param ARR_OUT The output array that will will be accessed.
 */
#define BLOCK_BEGIN_WITH_OUTPUT_TU(ARR_IN, ARR_OUT, EXTRA_PRAGMA_CLAUSE) \
    T *ARR_IN##_buffer = (T *) ARR_IN.data; \
    U *ARR_OUT##_buffer = (U *) ARR_OUT.data; \
    __attribute__((unused)) int64_t ARR_IN##_buffer_start = 0; \
    FOR_3D_BEGIN(ARR_IN, EXTRA_PRAGMA_CLAUSE) \
    int64_t flat_index = z*ARR_IN##_Ny*ARR_IN##_Nx + y*ARR_IN##_Nx + x;

/**
 * Closes the block started by `BLOCK_BEGIN_WITH_OUTPUT_TU`.
 */
#define BLOCK_END_WITH_OUTPUT_TU() FOR_3D_END()

#else

/**
 * Inserts boilerplate code for accessing the given parameter, `ARR`, in a blocked (chunked) manner.
 * For OpenACC, this macro will first start a blocked outer loop, push down the sizes and then start a flat inner loop.
 * For OpenMP, this macro will start a 3D loop.
 * This macro allows for additional pragma clauses, `EXTRA_PRAGMA_CLAUSE`, to be added to the parallel loop.
 * E.g. `BLOCK_BEGIN(ARR, reduction(+:sum))` will generate `#pragma acc parallel loop reduction(+:sum)` for OpenACC and `#pragma omp parallel for reduction(+:sum)` for OpenMP.
 *
 * @param ARR The array that will be accessed.
 * @param EXTRA_PRAGMA_CLAUSE Additional pragma clauses to be added to the parallel loop.
 */
#define BLOCK_BEGIN(ARR, EXTRA_PRAGMA_CLAUSE) \
    int64_t flat_index = 0; \
    ARR##_type *ARR##_buffer = (ARR##_type *) ARR.data; \
    __attribute__((unused)) int64_t ARR##_buffer_start = 0; \
    FOR_3D_BEGIN(ARR, EXTRA_PRAGMA_CLAUSE)

/**
 * Closes the block started by `BLOCK_BEGIN`.
 */
#define BLOCK_END() \
    flat_index++; \
    FOR_3D_END()

/**
 * Inserts boilerplate code for accessing the given parameter, `ARR`, in a blocked (chunked) manner, where the array has type `T`.
 * For OpenACC, this macro will first start a blocked outer loop, push down the sizes and then start a flat inner loop.
 * For OpenMP, this macro will start a 3D loop.
 * This macro allows for additional pragma clauses, `EXTRA_PRAGMA_CLAUSE`, to be added to the parallel loop.
 * E.g. `BLOCK_BEGIN_T(ARR, reduction(+:sum))` will generate `#pragma acc parallel loop reduction(+:sum)` for OpenACC and `#pragma omp parallel for reduction(+:sum)` for OpenMP.
 *
 * @param ARR The array that will be accessed.
 * @param EXTRA_PRAGMA_CLAUSE Additional pragma clauses to be added to the parallel loop.
 */
#define BLOCK_BEGIN_T(ARR, EXTRA_PRAGMA_CLAUSE) \
    int64_t flat_index = 0; \
    T *ARR##_buffer = (T *) ARR.data; \
    __attribute__((unused)) int64_t ARR##_buffer_start = 0; \
    FOR_3D_BEGIN(ARR, EXTRA_PRAGMA_CLAUSE)

/**
 * Closes the block started by `BLOCK_BEGIN_T`.
 */
#define BLOCK_END_T() \
    flat_index++; \
    FOR_3D_END()

/**
 * Inserts boilerplate code for accessing the given parameter, `ARR_IN`, in a blocked (chunked) manner, and also provides an output array, `ARR_OUT`, also in a blocked manner.
 * This macro assumes that the input and output arrays have the same length.
 * For OpenACC, this macro will first start a blocked outer loop, push down the sizes and then start a flat inner loop.
 * For OpenMP, this macro will start a 3D loop.
 * This macro allows for additional pragma clauses, `EXTRA_PRAGMA_CLAUSE`, to be added to the parallel loop.
 * E.g. `BLOCK_BEGIN_WITH_OUTPUT(ARR_IN, ARR_OUT, reduction(+:sum))` will generate `#pragma acc parallel loop reduction(+:sum)` for OpenACC and `#pragma omp parallel for reduction(+:sum)` for OpenMP.
 *
 * @param ARR_IN The input array that will be accessed.
 * @param ARR_OUT The output array that will will be accessed.
 * @param EXTRA_PRAGMA_CLAUSE Additional pragma clauses to be added to the parallel loop.
 */
#define BLOCK_BEGIN_WITH_OUTPUT(ARR_IN, ARR_OUT, EXTRA_PRAGMA_CLAUSE) \
    int64_t flat_index = 0; \
    ARR_IN##_type *ARR_IN##_buffer = (ARR_IN##_type *) ARR_IN.data; \
    ARR_OUT##_type *ARR_OUT##_buffer = (ARR_OUT##_type *) ARR_OUT.data; \
    __attribute__((unused)) int64_t ARR_IN##_buffer_start = 0; \
    FOR_3D_BEGIN(ARR_IN, EXTRA_PRAGMA_CLAUSE)

/**
 * Closes the block started by `BLOCK_BEGIN_WITH_OUTPUT`.
 */
#define BLOCK_END_WITH_OUTPUT() \
    flat_index++; \
    FOR_3D_END()

/**
 * Inserts boilerplate code for accessing the given parameter, `ARR_IN`, in a blocked (chunked) manner, and also provides an output array, `ARR_OUT`, also in a blocked manner.
 * This macro assumes that the input and output arrays have the same length.
 * This version is for templated types, where `ARR_IN` has type `T` and `ARR_OUT` has type `U`.
 * For OpenACC, this macro will first start a blocked outer loop, push down the sizes and then start a flat inner loop.
 * For OpenMP, this macro will start a 3D loop.
 * This macro allows for additional pragma clauses, `EXTRA_PRAGMA_CLAUSE`, to be added to the parallel loop.
 * E.g. `BLOCK_BEGIN_WITH_OUTPUT_TU(ARR_IN, ARR_OUT, reduction(+:sum))` will generate `#pragma acc parallel loop reduction(+:sum)` for OpenACC and `#pragma omp parallel for reduction(+:sum)` for OpenMP.
 *
 * @param ARR_IN The input array that will be accessed.
 * @param ARR_OUT The output array that will will be accessed.
 */
#define BLOCK_BEGIN_WITH_OUTPUT_TU(ARR_IN, ARR_OUT, EXTRA_PRAGMA_CLAUSE) \
    int64_t flat_index = 0; \
    T *ARR_IN##_buffer = (T *) ARR_IN.data; \
    U *ARR_OUT##_buffer = (U *) ARR_OUT.data; \
    __attribute__((unused)) int64_t ARR_IN##_buffer_start = 0; \
    FOR_3D_BEGIN(ARR_IN, EXTRA_PRAGMA_CLAUSE)

/**
 * Closes the block started by `BLOCK_BEGIN_WITH_OUTPUT_TU`.
 */
#define BLOCK_END_WITH_OUTPUT_TU() \
    flat_index++; \
    FOR_3D_END()

#endif
#endif

/**
 * Unpacks the sizes of the array `ARR` into the variables `ARR_Nz`, `ARR_Ny`, `ARR_Nx` and `ARR_length`.
 *
 * Following this call, the following variables will be exposed:
 *
 * - `ARR_Nz`: the size of the array in the z-dimension.
 *
 * - `ARR_Ny`: the size of the array in the y-dimension.
 *
 * - `ARR_Nx`: the size of the array in the x-dimension.
 *
 * - `ARR_length`: the total length of the array.
 *
 * @param ARR The array whose sizes will be unpacked.
 */
#define UNPACK_NUMPY(ARR) \
    ssize_t \
        __attribute__((unused)) ARR##_Nz = ARR.shape[0], \
        __attribute__((unused)) ARR##_Ny = ARR.shape[1], \
        __attribute__((unused)) ARR##_Nx = ARR.shape[2], \
        __attribute__((unused)) ARR##_length = ARR##_Nz*ARR##_Ny*ARR##_Nx;

#endif