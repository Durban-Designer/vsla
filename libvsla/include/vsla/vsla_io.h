/**
 * @file vsla_io.h
 * @brief Input/output operations for VSLA tensors
 * 
 * @copyright MIT License
 */

#ifndef VSLA_IO_H
#define VSLA_IO_H

#include "vsla_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Magic number for VSLA file format
 */
#define VSLA_FILE_MAGIC "VSLAv01\0"

/**
 * @brief Endianness values for file format
 */
typedef enum {
    VSLA_ENDIAN_LITTLE = 0,
    VSLA_ENDIAN_BIG = 1
} vsla_endian_t;

/**
 * @brief Save a tensor to a binary file
 * 
 * The file format is:
 * - 8 bytes: "VSLAv01\0" magic string
 * - 1 byte: endianness (0 = little, 1 = big)
 * - 1 byte: rank
 * - 1 byte: model
 * - 1 byte: dtype
 * - 4 bytes: reserved (set to 0)
 * - 8*rank bytes: shape array
 * - 8*rank bytes: cap array
 * - 8*rank bytes: stride array
 * - data bytes: tensor data in row-major order
 * 
 * @param path File path to save to
 * @param tensor Tensor to save
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_save(const char* path, const vsla_tensor_t* tensor);

/**
 * @brief Load a tensor from a binary file
 * 
 * @param path File path to load from
 * @param tensor Output tensor pointer (will be allocated)
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_load(const char* path, vsla_tensor_t** tensor);

/**
 * @brief Save tensor to file descriptor
 * 
 * @param fd File descriptor (must be open for writing)
 * @param tensor Tensor to save
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_save_fd(int fd, const vsla_tensor_t* tensor);

/**
 * @brief Load tensor from file descriptor
 * 
 * @param fd File descriptor (must be open for reading)
 * @param tensor Output tensor pointer (will be allocated)
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_load_fd(int fd, vsla_tensor_t** tensor);

/**
 * @brief Export tensor to CSV format (for debugging)
 * 
 * Only works for 1D and 2D tensors.
 * 
 * @param path File path to save to
 * @param tensor Tensor to export
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_export_csv(const char* path, const vsla_tensor_t* tensor);

/**
 * @brief Import tensor from CSV format
 * 
 * Creates a 2D tensor from CSV data.
 * 
 * @param path File path to load from
 * @param model Model type for created tensor
 * @param dtype Data type for created tensor
 * @param tensor Output tensor pointer (will be allocated)
 * @return VSLA_SUCCESS or error code
 */
vsla_error_t vsla_import_csv(const char* path, vsla_model_t model, 
                             vsla_dtype_t dtype, vsla_tensor_t** tensor);

/**
 * @brief Get system endianness
 * 
 * @return VSLA_ENDIAN_LITTLE or VSLA_ENDIAN_BIG
 */
vsla_endian_t vsla_get_endianness(void);

/**
 * @brief Swap byte order of a value
 * 
 * @param value Pointer to value
 * @param size Size in bytes
 */
void vsla_swap_bytes(void* value, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* VSLA_IO_H */