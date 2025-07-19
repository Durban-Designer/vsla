#include "test_framework.h"
#include "vsla/vsla_backend.h"
#include "vsla/vsla_tensor.h"

int test_backend_add(vsla_backend_interface_t* backend);

int main(void) {
    vsla_backend_interface_t* cpu_backend = vsla_backend_cpu_create();

    test_backend_add(cpu_backend);

    return 0;
}

int test_backend_add(vsla_backend_interface_t* backend) {
    uint64_t shape[] = {3};
    vsla_tensor_t* a = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* b = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
    vsla_tensor_t* out = vsla_new(1, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);

    backend->allocate(a);
    backend->allocate(b);
    backend->allocate(out);

    double a_data[] = {1.0, 2.0, 3.0};
    double b_data[] = {4.0, 5.0, 6.0};
    vsla_fill_basic(a, 0.0);
    vsla_fill_basic(b, 0.0);

    for(uint64_t i = 0; i < 3; ++i) {
        uint64_t idx[] = {i};
        vsla_set_f64_basic(a, idx, a_data[i]);
        vsla_set_f64_basic(b, idx, b_data[i]);
    }

    backend->add(out, a, b);

    double expected[] = {5.0, 7.0, 9.0};
    for(uint64_t i = 0; i < 3; ++i) {
        uint64_t idx[] = {i};
        double val;
        vsla_get_f64_basic(out, idx, &val);
        ASSERT_DOUBLE_EQ(val, expected[i], 1e-10);
    }

    vsla_free(a);
    vsla_free(b);
    vsla_free(out);
    
    return 1; // Test passed
}
