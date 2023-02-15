#ifndef boilerplate_h
#define boilerplate_h

// TODO it seems like vscode doesn't pick this up.
/// \def for_block_begin(arr)
/// Inserts boilerplate code for accessing \a arr in a blocked (chunked) manner.
#define for_block_begin(arr) \
    for (int64_t block_start = 0; block_start < arr##_length; block_start += acc_block_size<arr##_type>) { \
        const arr##_type *arr##_buffer = arr.data + block_start; \
        ssize_t arr##_buffer_length = min(acc_block_size<arr##_type>, arr##_length-block_start); \
        _Pragma(STR(acc data copyin(arr##_buffer[:arr##_buffer_length]))) \
        { \

#define for_block_end() } }

#define for_3d_begin(arr) \
    for (int64_t z = 0; z < arr##_Nz; z++) { \
        for (int64_t y = 0; y < arr##_Ny; y++) { \
            for (int64_t x = 0; x < arr##_Nx; x++) { \
                int64_t flat_index = z*arr##_Ny*arr##_Nx + y*arr##_Nx + x;

#define for_3d_end() }}}

#define for_flat_begin_1(arr) for_flat_begin(arr, arr)
#define for_flat_begin_2(arr, global_prefix) \
    for (int64_t flat_index = 0; flat_index < arr##_length; flat_index++) { \
        int64_t \
            global_prefix##_index = arr##_start + flat_index \
            z = global_prefix##_index / (arr##_Ny*arr##_Nx), \
            y = (global_prefix##_index / arr##_Nx) % arr##_Ny, \
            x = global_prefix##_index % arr##_Nx;

#define for_flat_end() }

// TODO I'm not sure this'll expand right.
#define for_flat_block_begin(arr) \
    for_block_begin(arr) \
    for_flat_begin_2(arr##_buffer, global)

#define for_flat_block_end() \
    for_flat_end() \
    for_block_end()

#define unpack_numpy(arr) \
    ssize_t \
        arr##_Nz = arr.shape[0], \
        arr##_Ny = arr.shape[1], \
        arr##_Nx = arr.shape[2], \
        arr##_length = arr##_Nz*arr##_Ny*arr##_Nx;

#endif