/**
 * @file vsla_io.c
 * @brief Input/output operations for VSLA tensors
 * 
 * @copyright MIT License
 */

#define _POSIX_C_SOURCE 200809L

#include "vsla/vsla_io.h"
#include "vsla/vsla_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>

static const char VSLA_MAGIC[] = "VSLAv01\0";
static const size_t VSLA_MAGIC_SIZE = 8;

vsla_endian_t vsla_get_endianness(void) {
    uint16_t test = 0x0001;
    uint8_t* bytes = (uint8_t*)&test;
    return bytes[0] == 0x01 ? VSLA_ENDIAN_LITTLE : VSLA_ENDIAN_BIG;
}

void vsla_swap_bytes(void* value, size_t size) {
    if (!value || size <= 1) return;
    
    uint8_t* bytes = (uint8_t*)value;
    for (size_t i = 0; i < size / 2; i++) {
        uint8_t temp = bytes[i];
        bytes[i] = bytes[size - 1 - i];
        bytes[size - 1 - i] = temp;
    }
}

static void swap_if_needed(void* data, size_t element_size, size_t count, 
                          vsla_endian_t file_endian) {
    vsla_endian_t system_endian = vsla_get_endianness();
    if (file_endian != system_endian && element_size > 1) {
        uint8_t* bytes = (uint8_t*)data;
        for (size_t i = 0; i < count; i++) {
            vsla_swap_bytes(bytes + i * element_size, element_size);
        }
    }
}

static size_t safe_write(int fd, const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    size_t total_written = 0;
    
    while (total_written < size) {
        ssize_t written = write(fd, bytes + total_written, size - total_written);
        if (written < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        total_written += written;
    }
    return total_written;
}

static size_t safe_read(int fd, void* data, size_t size) {
    uint8_t* bytes = (uint8_t*)data;
    size_t total_read = 0;
    
    while (total_read < size) {
        ssize_t bytes_read = read(fd, bytes + total_read, size - total_read);
        if (bytes_read < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (bytes_read == 0) break; // EOF
        total_read += bytes_read;
    }
    return total_read;
}

vsla_error_t vsla_save_fd(int fd, const vsla_tensor_t* tensor) {
    if (!tensor) return VSLA_ERROR_NULL_POINTER;
    if (fd < 0) return VSLA_ERROR_INVALID_ARGUMENT;
    
    vsla_endian_t endian = vsla_get_endianness();
    
    // Write magic number
    if (safe_write(fd, VSLA_MAGIC, VSLA_MAGIC_SIZE) != VSLA_MAGIC_SIZE) {
        return VSLA_ERROR_IO;
    }
    
    // Write header
    uint8_t header[4] = {
        (uint8_t)endian,
        tensor->rank,
        tensor->model,
        tensor->dtype
    };
    if (safe_write(fd, header, 4) != 4) return VSLA_ERROR_IO;
    
    // Write reserved bytes
    uint32_t reserved = 0;
    if (safe_write(fd, &reserved, 4) != 4) return VSLA_ERROR_IO;
    
    // Write shape, cap, and stride arrays
    size_t array_size = tensor->rank * sizeof(uint64_t);
    if (safe_write(fd, tensor->shape, array_size) != array_size) return VSLA_ERROR_IO;
    if (safe_write(fd, tensor->cap, array_size) != array_size) return VSLA_ERROR_IO;
    if (safe_write(fd, tensor->stride, array_size) != array_size) return VSLA_ERROR_IO;
    
    // Calculate data size
    uint64_t total_elements = 1;
    for (uint8_t i = 0; i < tensor->rank; i++) {
        total_elements *= tensor->cap[i];
    }
    
    size_t element_size = vsla_dtype_size((vsla_dtype_t)tensor->dtype);
    if (element_size == 0) return VSLA_ERROR_INVALID_DTYPE;
    
    size_t data_size = total_elements * element_size;
    
    // Write tensor data
    if (safe_write(fd, tensor->data, data_size) != data_size) {
        return VSLA_ERROR_IO;
    }
    
    return VSLA_SUCCESS;
}

vsla_error_t vsla_load_fd(int fd, vsla_tensor_t** tensor) {
    if (!tensor) return VSLA_ERROR_NULL_POINTER;
    if (fd < 0) return VSLA_ERROR_INVALID_ARGUMENT;
    
    *tensor = NULL;
    
    // Read and verify magic number
    char magic[VSLA_MAGIC_SIZE];
    if (safe_read(fd, magic, VSLA_MAGIC_SIZE) != VSLA_MAGIC_SIZE) {
        return VSLA_ERROR_IO;
    }
    if (memcmp(magic, VSLA_MAGIC, VSLA_MAGIC_SIZE) != 0) {
        return VSLA_ERROR_INVALID_FILE;
    }
    
    // Read header
    uint8_t header[4];
    if (safe_read(fd, header, 4) != 4) return VSLA_ERROR_IO;
    
    vsla_endian_t file_endian = (vsla_endian_t)header[0];
    uint8_t rank = header[1];
    uint8_t model = header[2];
    uint8_t dtype = header[3];
    
    // Validate header values
    if (rank == 0) return VSLA_ERROR_INVALID_RANK;
    if (model > VSLA_MODEL_B) return VSLA_ERROR_INVALID_MODEL;
    if (dtype > VSLA_DTYPE_F32) return VSLA_ERROR_INVALID_DTYPE;
    
    // Skip reserved bytes
    uint32_t reserved;
    if (safe_read(fd, &reserved, 4) != 4) return VSLA_ERROR_IO;
    
    // Allocate arrays
    size_t array_size = rank * sizeof(uint64_t);
    uint64_t* shape = malloc(array_size);
    uint64_t* cap = malloc(array_size);
    uint64_t* stride = malloc(array_size);
    
    if (!shape || !cap || !stride) {
        free(shape);
        free(cap);
        free(stride);
        return VSLA_ERROR_MEMORY;
    }
    
    // Read arrays
    if (safe_read(fd, shape, array_size) != array_size ||
        safe_read(fd, cap, array_size) != array_size ||
        safe_read(fd, stride, array_size) != array_size) {
        free(shape);
        free(cap);
        free(stride);
        return VSLA_ERROR_IO;
    }
    
    // Swap byte order if needed
    swap_if_needed(shape, sizeof(uint64_t), rank, file_endian);
    swap_if_needed(cap, sizeof(uint64_t), rank, file_endian);
    swap_if_needed(stride, sizeof(uint64_t), rank, file_endian);
    
    // Calculate data size and allocate tensor
    uint64_t total_elements = 1;
    for (uint8_t i = 0; i < rank; i++) {
        if (cap[i] == 0) {
            free(shape);
            free(cap);
            free(stride);
            return VSLA_ERROR_INVALID_FILE;
        }
        total_elements *= cap[i];
    }
    
    size_t element_size = vsla_dtype_size((vsla_dtype_t)dtype);
    size_t data_size = total_elements * element_size;
    
    // Allocate data buffer with 64-byte alignment
    void* data;
    if (posix_memalign(&data, 64, data_size) != 0) {
        free(shape);
        free(cap);
        free(stride);
        return VSLA_ERROR_MEMORY;
    }
    
    // Read tensor data
    if (safe_read(fd, data, data_size) != data_size) {
        free(shape);
        free(cap);
        free(stride);
        free(data);
        return VSLA_ERROR_IO;
    }
    
    // Swap data byte order if needed
    swap_if_needed(data, element_size, total_elements, file_endian);
    
    // Create tensor structure
    vsla_tensor_t* t = malloc(sizeof(vsla_tensor_t));
    if (!t) {
        free(shape);
        free(cap);
        free(stride);
        free(data);
        return VSLA_ERROR_MEMORY;
    }
    
    t->rank = rank;
    t->model = model;
    t->dtype = dtype;
    t->flags = 0;
    t->shape = shape;
    t->cap = cap;
    t->stride = stride;
    t->data = data;
    
    *tensor = t;
    return VSLA_SUCCESS;
}

vsla_error_t vsla_save(const char* path, const vsla_tensor_t* tensor) {
    if (!path || !tensor) return VSLA_ERROR_NULL_POINTER;
    
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return VSLA_ERROR_IO;
    
    vsla_error_t result = vsla_save_fd(fd, tensor);
    close(fd);
    return result;
}

vsla_error_t vsla_load(const char* path, vsla_tensor_t** tensor) {
    if (!path || !tensor) return VSLA_ERROR_NULL_POINTER;
    
    int fd = open(path, O_RDONLY);
    if (fd < 0) return VSLA_ERROR_IO;
    
    vsla_error_t result = vsla_load_fd(fd, tensor);
    close(fd);
    return result;
}

vsla_error_t vsla_export_csv(const char* path, const vsla_tensor_t* tensor) {
    if (!path || !tensor) return VSLA_ERROR_NULL_POINTER;
    if (tensor->rank > 2) return VSLA_ERROR_INVALID_ARGUMENT;
    
    FILE* file = fopen(path, "w");
    if (!file) return VSLA_ERROR_IO;
    
    size_t element_size = vsla_dtype_size((vsla_dtype_t)tensor->dtype);
    if (element_size == 0) {
        fclose(file);
        return VSLA_ERROR_INVALID_DTYPE;
    }
    
    if (tensor->rank == 1) {
        // Export 1D tensor as a single row
        for (uint64_t i = 0; i < tensor->shape[0]; i++) {
            double value = 0.0;
            vsla_error_t err = vsla_get_f64(tensor, &i, &value);
            if (err != VSLA_SUCCESS) {
                fclose(file);
                return err;
            }
            fprintf(file, "%.15g", value);
            if (i < tensor->shape[0] - 1) fprintf(file, ",");
        }
        fprintf(file, "\n");
    } else {
        // Export 2D tensor
        for (uint64_t i = 0; i < tensor->shape[0]; i++) {
            for (uint64_t j = 0; j < tensor->shape[1]; j++) {
                uint64_t indices[2] = {i, j};
                double value = 0.0;
                vsla_error_t err = vsla_get_f64(tensor, indices, &value);
                if (err != VSLA_SUCCESS) {
                    fclose(file);
                    return err;
                }
                fprintf(file, "%.15g", value);
                if (j < tensor->shape[1] - 1) fprintf(file, ",");
            }
            fprintf(file, "\n");
        }
    }
    
    fclose(file);
    return VSLA_SUCCESS;
}

vsla_error_t vsla_import_csv(const char* path, vsla_model_t model, 
                             vsla_dtype_t dtype, vsla_tensor_t** tensor) {
    if (!path || !tensor) return VSLA_ERROR_NULL_POINTER;
    
    FILE* file = fopen(path, "r");
    if (!file) return VSLA_ERROR_IO;
    
    // Count rows and columns
    uint64_t rows = 0, cols = 0;
    char line[4096];
    
    while (fgets(line, sizeof(line), file)) {
        if (rows == 0) {
            // Count columns in first row
            char* token = strtok(line, ",");
            while (token) {
                cols++;
                token = strtok(NULL, ",");
            }
        }
        rows++;
    }
    
    if (rows == 0 || cols == 0) {
        fclose(file);
        return VSLA_ERROR_INVALID_FILE;
    }
    
    // Create tensor
    uint64_t shape[2] = {rows, cols};
    *tensor = vsla_new(2, shape, model, dtype);
    if (!*tensor) {
        fclose(file);
        return VSLA_ERROR_MEMORY;
    }
    
    // Read data
    rewind(file);
    uint64_t row = 0;
    
    while (fgets(line, sizeof(line), file) && row < rows) {
        char* token = strtok(line, ",");
        uint64_t col = 0;
        
        while (token && col < cols) {
            double value = strtod(token, NULL);
            uint64_t indices[2] = {row, col};
            vsla_error_t err = vsla_set_f64(*tensor, indices, value);
            if (err != VSLA_SUCCESS) {
                fclose(file);
                vsla_free(*tensor);
                *tensor = NULL;
                return err;
            }
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }
    
    fclose(file);
    return VSLA_SUCCESS;
}