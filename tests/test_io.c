/**
 * @file test_io.c
 * @brief Unit tests for VSLA I/O operations
 * 
 * @copyright MIT License
 */

#include "test_framework.h"
#include "vsla/vsla.h"
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

// Test data paths
static const char* TEST_BINARY_PATH = "/tmp/test_tensor.vsla";
static const char* TEST_CSV_PATH = "/tmp/test_tensor.csv";

static void cleanup_test_files(void) {
    unlink(TEST_BINARY_PATH);
    unlink(TEST_CSV_PATH);
}

// Test endianness detection
static int test_endianness(void) {
    vsla_endian_t endian = vsla_get_endianness();
    return (endian == VSLA_ENDIAN_LITTLE || endian == VSLA_ENDIAN_BIG);
}

// Test byte swapping
static int test_byte_swapping(void) {
    uint32_t value = 0x12345678;
    uint32_t original = value;
    
    vsla_swap_bytes(&value, sizeof(value));
    if (value == 0x78563412) {
        vsla_swap_bytes(&value, sizeof(value));
        return value == original;
    }
    return 0;
}

// Test saving and loading a simple tensor
static int test_binary_save_load_simple(void) {
    cleanup_test_files();
    
    // Create test tensor
    uint64_t shape[] = {3, 2};
    vsla_tensor_t* tensor = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (!tensor) return 0;
    
    // Fill with test data
    for (uint64_t i = 0; i < shape[0]; i++) {
        for (uint64_t j = 0; j < shape[1]; j++) {
            uint64_t indices[] = {i, j};
            double value = (double)(i * shape[1] + j + 1);
            if (vsla_set_f64(tensor, indices, value) != VSLA_SUCCESS) {
                vsla_free(tensor);
                return 0;
            }
        }
    }
    
    // Save tensor
    vsla_error_t err = vsla_save(TEST_BINARY_PATH, tensor);
    if (err != VSLA_SUCCESS) {
        vsla_free(tensor);
        return 0;
    }
    
    // Load tensor
    vsla_tensor_t* loaded = NULL;
    err = vsla_load(TEST_BINARY_PATH, &loaded);
    if (err != VSLA_SUCCESS || !loaded) {
        vsla_free(tensor);
        return 0;
    }
    
    // Verify properties
    int success = 1;
    if (loaded->rank != tensor->rank ||
        loaded->model != tensor->model ||
        loaded->dtype != tensor->dtype) {
        success = 0;
    }
    
    // Verify shape
    for (uint8_t i = 0; i < tensor->rank && success; i++) {
        if (loaded->shape[i] != tensor->shape[i]) {
            success = 0;
        }
    }
    
    // Verify data
    for (uint64_t i = 0; i < shape[0] && success; i++) {
        for (uint64_t j = 0; j < shape[1] && success; j++) {
            uint64_t indices[] = {i, j};
            double orig_val, loaded_val;
            if (vsla_get_f64(tensor, indices, &orig_val) != VSLA_SUCCESS ||
                vsla_get_f64(loaded, indices, &loaded_val) != VSLA_SUCCESS ||
                fabs(orig_val - loaded_val) > 1e-15) {
                success = 0;
            }
        }
    }
    
    vsla_free(tensor);
    vsla_free(loaded);
    cleanup_test_files();
    return success;
}

// Test saving and loading 1D tensor
static int test_binary_save_load_1d(void) {
    cleanup_test_files();
    
    uint64_t shape[] = {5};
    vsla_tensor_t* tensor = vsla_new(1, shape, VSLA_MODEL_B, VSLA_DTYPE_F32);
    if (!tensor) return 0;
    
    // Fill with test data
    for (uint64_t i = 0; i < shape[0]; i++) {
        double value = (double)i * 1.5 + 0.5;
        if (vsla_set_f64(tensor, &i, value) != VSLA_SUCCESS) {
            vsla_free(tensor);
            return 0;
        }
    }
    
    // Save and load
    if (vsla_save(TEST_BINARY_PATH, tensor) != VSLA_SUCCESS) {
        vsla_free(tensor);
        return 0;
    }
    
    vsla_tensor_t* loaded = NULL;
    if (vsla_load(TEST_BINARY_PATH, &loaded) != VSLA_SUCCESS || !loaded) {
        vsla_free(tensor);
        return 0;
    }
    
    // Verify
    int success = (loaded->rank == 1 && 
                   loaded->model == VSLA_MODEL_B &&
                   loaded->dtype == VSLA_DTYPE_F32 &&
                   loaded->shape[0] == 5);
    
    if (success) {
        for (uint64_t i = 0; i < shape[0]; i++) {
            double orig_val, loaded_val;
            if (vsla_get_f64(tensor, &i, &orig_val) != VSLA_SUCCESS ||
                vsla_get_f64(loaded, &i, &loaded_val) != VSLA_SUCCESS ||
                fabs(orig_val - loaded_val) > 1e-6) {  // F32 precision
                success = 0;
                break;
            }
        }
    }
    
    vsla_free(tensor);
    vsla_free(loaded);
    cleanup_test_files();
    return success;
}

// Test file descriptor operations
static int test_fd_operations(void) {
    cleanup_test_files();
    
    uint64_t shape[] = {2, 3};
    vsla_tensor_t* tensor = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (!tensor) return 0;
    
    // Fill with test data
    for (uint64_t i = 0; i < shape[0]; i++) {
        for (uint64_t j = 0; j < shape[1]; j++) {
            uint64_t indices[] = {i, j};
            double value = (double)(i + j + 1) * 0.1;
            if (vsla_set_f64(tensor, indices, value) != VSLA_SUCCESS) {
                vsla_free(tensor);
                return 0;
            }
        }
    }
    
    // Save using file descriptor
    int fd = open(TEST_BINARY_PATH, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        vsla_free(tensor);
        return 0;
    }
    
    if (vsla_save_fd(fd, tensor) != VSLA_SUCCESS) {
        close(fd);
        vsla_free(tensor);
        return 0;
    }
    close(fd);
    
    // Load using file descriptor
    fd = open(TEST_BINARY_PATH, O_RDONLY);
    if (fd < 0) {
        vsla_free(tensor);
        return 0;
    }
    
    vsla_tensor_t* loaded = NULL;
    if (vsla_load_fd(fd, &loaded) != VSLA_SUCCESS || !loaded) {
        close(fd);
        vsla_free(tensor);
        return 0;
    }
    close(fd);
    
    // Verify data
    int success = 1;
    for (uint64_t i = 0; i < shape[0] && success; i++) {
        for (uint64_t j = 0; j < shape[1] && success; j++) {
            uint64_t indices[] = {i, j};
            double orig_val, loaded_val;
            if (vsla_get_f64(tensor, indices, &orig_val) != VSLA_SUCCESS ||
                vsla_get_f64(loaded, indices, &loaded_val) != VSLA_SUCCESS ||
                fabs(orig_val - loaded_val) > 1e-15) {
                success = 0;
            }
        }
    }
    
    vsla_free(tensor);
    vsla_free(loaded);
    cleanup_test_files();
    return success;
}

// Test CSV export/import for 1D tensor
static int test_csv_1d(void) {
    cleanup_test_files();
    
    uint64_t shape[] = {4};
    vsla_tensor_t* tensor = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (!tensor) return 0;
    
    double test_values[] = {1.5, -2.0, 3.14159, 0.0};
    for (uint64_t i = 0; i < shape[0]; i++) {
        if (vsla_set_f64(tensor, &i, test_values[i]) != VSLA_SUCCESS) {
            vsla_free(tensor);
            return 0;
        }
    }
    
    // Export to CSV
    if (vsla_export_csv(TEST_CSV_PATH, tensor) != VSLA_SUCCESS) {
        vsla_free(tensor);
        return 0;
    }
    
    // Import from CSV
    vsla_tensor_t* imported = NULL;
    if (vsla_import_csv(TEST_CSV_PATH, VSLA_MODEL_A, VSLA_DTYPE_F64, &imported) != VSLA_SUCCESS || !imported) {
        vsla_free(tensor);
        return 0;
    }
    
    // Verify: CSV import creates 2D tensor (1 row, N columns)
    int success = (imported->rank == 2 && 
                   imported->shape[0] == 1 && 
                   imported->shape[1] == 4);
    
    if (success) {
        for (uint64_t j = 0; j < shape[0]; j++) {
            uint64_t indices[] = {0, j};
            double imported_val;
            if (vsla_get_f64(imported, indices, &imported_val) != VSLA_SUCCESS ||
                fabs(test_values[j] - imported_val) > 1e-14) {
                success = 0;
                break;
            }
        }
    }
    
    vsla_free(tensor);
    vsla_free(imported);
    cleanup_test_files();
    return success;
}

// Test CSV export/import for 2D tensor
static int test_csv_2d(void) {
    cleanup_test_files();
    
    uint64_t shape[] = {2, 3};
    vsla_tensor_t* tensor = vsla_new(2, shape, VSLA_MODEL_B, VSLA_DTYPE_F32);
    if (!tensor) return 0;
    
    double test_data[2][3] = {
        {1.0, 2.5, -1.5},
        {0.0, -3.14, 42.0}
    };
    
    for (uint64_t i = 0; i < shape[0]; i++) {
        for (uint64_t j = 0; j < shape[1]; j++) {
            uint64_t indices[] = {i, j};
            if (vsla_set_f64(tensor, indices, test_data[i][j]) != VSLA_SUCCESS) {
                vsla_free(tensor);
                return 0;
            }
        }
    }
    
    // Export to CSV
    if (vsla_export_csv(TEST_CSV_PATH, tensor) != VSLA_SUCCESS) {
        vsla_free(tensor);
        return 0;
    }
    
    // Import from CSV
    vsla_tensor_t* imported = NULL;
    if (vsla_import_csv(TEST_CSV_PATH, VSLA_MODEL_B, VSLA_DTYPE_F32, &imported) != VSLA_SUCCESS || !imported) {
        vsla_free(tensor);
        return 0;
    }
    
    // Verify
    int success = (imported->rank == 2 && 
                   imported->shape[0] == 2 && 
                   imported->shape[1] == 3 &&
                   imported->model == VSLA_MODEL_B &&
                   imported->dtype == VSLA_DTYPE_F32);
    
    if (success) {
        for (uint64_t i = 0; i < shape[0]; i++) {
            for (uint64_t j = 0; j < shape[1]; j++) {
                uint64_t indices[] = {i, j};
                double imported_val;
                if (vsla_get_f64(imported, indices, &imported_val) != VSLA_SUCCESS ||
                    fabs(test_data[i][j] - imported_val) > 1e-6) {  // F32 precision
                    success = 0;
                    break;
                }
            }
            if (!success) break;
        }
    }
    
    vsla_free(tensor);
    vsla_free(imported);
    cleanup_test_files();
    return success;
}

// Test error handling
static int test_error_handling(void) {
    cleanup_test_files();
    
    // Test NULL pointer errors
    if (vsla_save(NULL, NULL) != VSLA_ERROR_NULL_POINTER) return 0;
    if (vsla_load(NULL, NULL) != VSLA_ERROR_NULL_POINTER) return 0;
    if (vsla_save_fd(-1, NULL) != VSLA_ERROR_NULL_POINTER) return 0;
    if (vsla_load_fd(-1, NULL) != VSLA_ERROR_NULL_POINTER) return 0;
    if (vsla_export_csv(NULL, NULL) != VSLA_ERROR_NULL_POINTER) return 0;
    if (vsla_import_csv(NULL, VSLA_MODEL_A, VSLA_DTYPE_F64, NULL) != VSLA_ERROR_NULL_POINTER) return 0;
    
    // Test invalid file operations
    vsla_tensor_t* dummy = NULL;
    if (vsla_load("/nonexistent/path", &dummy) != VSLA_ERROR_IO) return 0;
    if (vsla_save("/invalid\0path", NULL) != VSLA_ERROR_NULL_POINTER) return 0;
    
    // Test invalid tensor for CSV export (rank > 2)
    uint64_t shape[] = {2, 2, 2};
    vsla_tensor_t* tensor3d = vsla_new(3, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    if (tensor3d) {
        if (vsla_export_csv(TEST_CSV_PATH, tensor3d) != VSLA_ERROR_INVALID_ARGUMENT) {
            vsla_free(tensor3d);
            return 0;
        }
        vsla_free(tensor3d);
    }
    
    return 1;
}

// Test file format validation
static int test_file_format_validation(void) {
    cleanup_test_files();
    
    // Create invalid file
    FILE* file = fopen(TEST_BINARY_PATH, "wb");
    if (!file) return 0;
    
    // Write invalid magic number
    const char* wrong_magic = "INVALID\0";
    fwrite(wrong_magic, 1, 8, file);
    fclose(file);
    
    // Try to load - should fail
    vsla_tensor_t* tensor = NULL;
    if (vsla_load(TEST_BINARY_PATH, &tensor) != VSLA_ERROR_INVALID_FILE) {
        return 0;
    }
    
    cleanup_test_files();
    return 1;
}

static void io_test_setup(void) {
    // Setup for I/O tests
}

static void io_test_teardown(void) {
    // Teardown for I/O tests - cleanup any remaining test files
    cleanup_test_files();
}

static void run_io_tests(void) {
    printf("Running I/O tests:\n");
    
    RUN_TEST(test_endianness);
    RUN_TEST(test_byte_swapping);
    RUN_TEST(test_binary_save_load_simple);
    RUN_TEST(test_binary_save_load_1d);
    RUN_TEST(test_fd_operations);
    RUN_TEST(test_csv_1d);
    RUN_TEST(test_csv_2d);
    RUN_TEST(test_error_handling);
    RUN_TEST(test_file_format_validation);
}

static const test_suite_t io_suite = {
    .name = "io",
    .setup = io_test_setup,
    .teardown = io_test_teardown,
    .run_tests = run_io_tests
};

void register_io_tests(void) {
    register_test_suite(&io_suite);
}