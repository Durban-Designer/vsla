/**
 * @file test_io.c
 * @brief Tests for tensor I/O operations
 * 
 * @copyright MIT License
 */

#include "test_framework.h"
#include "vsla/vsla.h"
#include "vsla/vsla_io.h"
#include <unistd.h>
#include <fcntl.h>

#define TEST_BINARY_PATH "test_tensor.bin"
#define TEST_CSV_PATH "test_tensor.csv"

static int test_endianness(void) {
    vsla_endian_t endian = vsla_get_endianness();
    uint16_t test_val = 0x0102;
    uint8_t* bytes = (uint8_t*)&test_val;
    if (endian == VSLA_ENDIAN_LITTLE) {
        ASSERT_EQ(bytes[0], 0x02);
    } else {
        ASSERT_EQ(bytes[0], 0x01);
    }

    uint32_t value = 0x12345678;
    uint32_t original_value = value;
    vsla_swap_bytes(&value, sizeof(value));
    ASSERT_NE(original_value, value);
    vsla_swap_bytes(&value, sizeof(value));
    ASSERT_EQ(original_value, value);
    return 1;
}

static int test_binary_save_load(void) {
    uint64_t shape[] = {2, 3};
    vsla_tensor_t* tensor = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    ASSERT_NOT_NULL(tensor);

    for (uint64_t i = 0; i < 6; ++i) {
        ((double*)tensor->data)[i] = (double)i;
    }

    vsla_error_t err = vsla_save(TEST_BINARY_PATH, tensor);
    ASSERT_EQ(VSLA_SUCCESS, err);

    vsla_tensor_t* loaded = NULL;
    err = vsla_load(TEST_BINARY_PATH, &loaded);
    ASSERT_EQ(VSLA_SUCCESS, err);
    ASSERT_NOT_NULL(loaded);

    ASSERT_EQ(tensor->rank, loaded->rank);
    ASSERT_EQ(tensor->model, loaded->model);
    ASSERT_EQ(tensor->dtype, loaded->dtype);
    for (uint8_t i = 0; i < tensor->rank; ++i) {
        ASSERT_EQ(tensor->shape[i], loaded->shape[i]);
    }

    for (uint64_t i = 0; i < 6; ++i) {
        ASSERT_FLOAT_EQ(((double*)tensor->data)[i], ((double*)loaded->data)[i], 1e-9);
    }

    vsla_free(tensor);
    vsla_free(loaded);
    remove(TEST_BINARY_PATH);
    return 1;
}

static void run_io_tests(void) {
    TEST_CASE("Endianness Detection and Swapping", test_endianness);
    TEST_CASE("Binary Save and Load", test_binary_save_load);
}

static const test_suite_t io_suite = {
    .name = "io",
    .setup = NULL,
    .teardown = NULL,
    .run_tests = run_io_tests
};

void register_io_tests(void) {
    register_test_suite(&io_suite);
}