/**
 * @file bindings_working.cpp
 * @brief Working Python bindings for VSLA using only available functions
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>

extern "C" {
#include "vsla/vsla.h"
}

namespace py = pybind11;

// Simplified wrapper using only available C functions
class PyVslaTensor {
public:
    PyVslaTensor() : ctx_(nullptr), tensor_(nullptr) {}
    
    // Constructor from numpy array (multi-dimensional support)
    PyVslaTensor(vsla_context_t* ctx, py::array_t<double> data, vsla_model_t model = VSLA_MODEL_A) : ctx_(ctx) {
        auto buf = data.request();

        if (buf.ndim > 8) {
            throw std::runtime_error("Maximum 8 dimensions supported");
        }

        // Convert shape to VSLA format
        std::vector<uint64_t> shape(buf.ndim);
        size_t total_elements = 1;
        for (int i = 0; i < buf.ndim; i++) {
            shape[i] = static_cast<uint64_t>(buf.shape[i]);
            total_elements *= shape[i];
        }

        tensor_ = vsla_tensor_create(ctx_, static_cast<uint8_t>(buf.ndim), shape.data(), model, VSLA_DTYPE_F64);

        if (!tensor_) {
            throw std::runtime_error("Failed to create VSLA tensor");
        }

        // Copy data
        double* src_ptr = static_cast<double*>(buf.ptr);
        size_t data_size;
        double* dst_ptr = static_cast<double*>(vsla_tensor_data_mut(tensor_, &data_size));
        if (dst_ptr && data_size >= total_elements * sizeof(double)) {
            std::memcpy(dst_ptr, src_ptr, total_elements * sizeof(double));
        }
    }
    
    // Move constructor (takes ownership)
    PyVslaTensor(vsla_context_t* ctx, vsla_tensor_t* tensor) : ctx_(ctx), tensor_(tensor) {}
    
    // Disable copy operations
    PyVslaTensor(const PyVslaTensor&) = delete;
    PyVslaTensor& operator=(const PyVslaTensor&) = delete;
    
    // Move constructor
    PyVslaTensor(PyVslaTensor&& other) noexcept : ctx_(other.ctx_), tensor_(other.tensor_) {
        other.ctx_ = nullptr;
        other.tensor_ = nullptr;
    }
    
    // Move assignment
    PyVslaTensor& operator=(PyVslaTensor&& other) noexcept {
        if (this != &other) {
            if (tensor_) {
                vsla_tensor_free(tensor_);
            }
            ctx_ = other.ctx_;
            tensor_ = other.tensor_;
            other.ctx_ = nullptr;
            other.tensor_ = nullptr;
        }
        return *this;
    }
    
    ~PyVslaTensor() {
        if (tensor_) {
            vsla_tensor_free(tensor_);
        }
    }
    
    // Convert back to numpy array
    py::array_t<double> to_numpy() const {
        if (!tensor_) {
            throw std::runtime_error("Invalid tensor");
        }

        uint8_t rank = vsla_get_rank(tensor_);
        std::vector<uint64_t> shape(rank);
        vsla_get_shape(tensor_, shape.data());
        
        // Convert to numpy shape format
        std::vector<py::ssize_t> np_shape(rank);
        size_t total_elements = 1;
        for (int i = 0; i < rank; i++) {
            np_shape[i] = static_cast<py::ssize_t>(shape[i]);
            total_elements *= shape[i];
        }
        
        auto result = py::array_t<double>(np_shape);
        auto buf = result.request();
        double* dst_ptr = static_cast<double*>(buf.ptr);
        
        size_t data_size;
        const double* src_ptr = static_cast<const double*>(vsla_tensor_data(tensor_, &data_size));
        if (src_ptr && data_size >= total_elements * sizeof(double)) {
            std::memcpy(dst_ptr, src_ptr, total_elements * sizeof(double));
        }

        return result;
    }

    // Addition operation
    PyVslaTensor add(const PyVslaTensor& other) const {
        if (!tensor_ || !other.tensor_) {
            throw std::runtime_error("Invalid tensor for addition");
        }

        uint8_t rank_a = vsla_get_rank(tensor_);
        uint8_t rank_b = vsla_get_rank(other.tensor_);
        
        // For now, require same rank
        if (rank_a != rank_b) {
            throw std::runtime_error("Addition requires same rank tensors for now");
        }
        
        std::vector<uint64_t> shape_a(rank_a);
        std::vector<uint64_t> shape_b(rank_b);
        vsla_get_shape(tensor_, shape_a.data());
        vsla_get_shape(other.tensor_, shape_b.data());
        
        // Calculate result shape (maximum in each dimension)
        std::vector<uint64_t> result_shape(rank_a);
        for (int i = 0; i < rank_a; i++) {
            result_shape[i] = std::max(shape_a[i], shape_b[i]);
        }
        
        vsla_tensor_t* result = vsla_tensor_create(ctx_, rank_a, result_shape.data(), 
                                                   vsla_get_model(tensor_), VSLA_DTYPE_F64);

        if (!result) {
            throw std::runtime_error("Failed to create result tensor");
        }

        vsla_error_t err = vsla_add(ctx_, result, tensor_, other.tensor_);
        if (err != VSLA_SUCCESS) {
            vsla_tensor_free(result);
            throw std::runtime_error("Addition failed: " + std::string(vsla_error_string(err)));
        }

        return PyVslaTensor(ctx_, result);
    }

    // Convolution operation (Model A only)
    PyVslaTensor convolve(const PyVslaTensor& other) const {
        if (!tensor_ || !other.tensor_) {
            throw std::runtime_error("Invalid tensor for convolution");
        }

        if (vsla_get_model(tensor_) != VSLA_MODEL_A || vsla_get_model(other.tensor_) != VSLA_MODEL_A) {
            throw std::runtime_error("Convolution only available in Model A");
        }

        uint8_t rank_a = vsla_get_rank(tensor_);
        uint8_t rank_b = vsla_get_rank(other.tensor_);
        
        if (rank_a != 1 || rank_b != 1) {
            throw std::runtime_error("Convolution currently only supports 1D tensors");
        }
        
        std::vector<uint64_t> shape_a(rank_a);
        std::vector<uint64_t> shape_b(rank_b);
        vsla_get_shape(tensor_, shape_a.data());
        vsla_get_shape(other.tensor_, shape_b.data());
        
        uint64_t result_size = shape_a[0] + shape_b[0] - 1;
        uint64_t result_shape[] = {result_size};
        vsla_tensor_t* result = vsla_tensor_create(ctx_, 1, result_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);

        if (!result) {
            throw std::runtime_error("Failed to create result tensor");
        }

        vsla_error_t err = vsla_conv(ctx_, result, tensor_, other.tensor_);
        if (err != VSLA_SUCCESS) {
            vsla_tensor_free(result);
            throw std::runtime_error("Convolution failed: " + std::string(vsla_error_string(err)));
        }

        return PyVslaTensor(ctx_, result);
    }
    
    // Get shape
    std::vector<uint64_t> shape() const {
        if (!tensor_) {
            return {};
        }
        uint8_t rank = vsla_get_rank(tensor_);
        std::vector<uint64_t> shape(rank);
        vsla_get_shape(tensor_, shape.data());
        return shape;
    }

    // Get model
    vsla_model_t model() const {
        if (!tensor_) {
            return VSLA_MODEL_A;
        }
        return vsla_get_model(tensor_);
    }

private:
    vsla_context_t* ctx_;
    vsla_tensor_t* tensor_;
};

// Global context (simplified)
static vsla_context_t* g_ctx = nullptr;

PYBIND11_MODULE(_core, m) {
    m.doc() = "VSLA: Variable-Shape Linear Algebra - Working Core C++ bindings";

    // Initialize VSLA context
    vsla_config_t config = {};
    config.backend = VSLA_BACKEND_AUTO;
    config.device_id = 0;
    config.memory_limit = 0;
    config.optimization_hint = VSLA_HINT_NONE;
    config.enable_profiling = false;
    config.verbose = false;
    g_ctx = vsla_init(&config);
    if (!g_ctx) {
        throw std::runtime_error("Failed to initialize VSLA library");
    }

    // Register cleanup function
    auto cleanup = []() {
        if (g_ctx) {
            vsla_cleanup(g_ctx);
            g_ctx = nullptr;
        }
    };
    m.add_object("_cleanup", py::capsule(cleanup));

    // Expose model enum
    py::enum_<vsla_model_t>(m, "Model")
        .value("A", VSLA_MODEL_A, "Convolution semiring model")
        .value("B", VSLA_MODEL_B, "Kronecker semiring model");

    // Expose main tensor class
    py::class_<PyVslaTensor>(m, "Tensor")
        .def(py::init([](py::array_t<double> data, vsla_model_t model) {
            return new PyVslaTensor(g_ctx, data, model);
        }), py::arg("data"), py::arg("model") = VSLA_MODEL_A,
             "Create a VSLA tensor from numpy array")
        .def("to_numpy", &PyVslaTensor::to_numpy,
             "Convert tensor to numpy array")
        .def("__add__", &PyVslaTensor::add,
             "Add two tensors with automatic shape promotion")
        .def("add", &PyVslaTensor::add,
             "Add two tensors with automatic shape promotion")
        .def("convolve", &PyVslaTensor::convolve,
             "Convolve with another tensor (Model A only)")
        .def("shape", &PyVslaTensor::shape,
             "Get tensor shape")
        .def("model", &PyVslaTensor::model,
             "Get semiring model ('A' or 'B')")
        .def("__repr__", [](const PyVslaTensor& t) {
            auto shape = t.shape();
            std::string shape_str = "(";
            for (size_t i = 0; i < shape.size(); i++) {
                if (i > 0) shape_str += ", ";
                shape_str += std::to_string(shape[i]);
            }
            shape_str += ")";
            return "VslaTensor(shape=" + shape_str + ", model=" + (t.model() == VSLA_MODEL_A ? "A" : "B") + ")";
        });

    // Version info
    m.attr("__version__") = "0.1.0";
}