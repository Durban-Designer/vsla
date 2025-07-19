#include "test_framework.h"

vsla_backend_interface_t* get_backend_by_name(const char* name) {
    if (strcmp(name, "CPU") == 0) {
        return vsla_backend_cpu_create();
    } else if (strcmp(name, "CUDA") == 0) {
        // CUDA backend not implemented yet
        return NULL;
    }
    return NULL;
}
