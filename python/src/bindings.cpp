/**
 * @file bindings.cpp
 * @brief Python bindings for VSLA using pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

extern "C" {
#include "vsla/vsla.h"
}

namespace py = pybind11;

// Wrapper class for VSLA tensor to manage memory properly
class PyVslaTensor {
public:
    PyVslaTensor(py::array_t<double> data, vsla_model_t model = VSLA_MODEL_A) {
        auto buf = data.request();
        
        if (buf.ndim != 1) {
            throw std::runtime_error("Only 1D arrays supported currently");
        }
        
        size_t size = buf.shape[0];
        tensor_ = vsla_new(1, &size, model, VSLA_DTYPE_F64);
        
        if (!tensor_) {
            throw std::runtime_error("Failed to create VSLA tensor");
        }
        
        // Copy data
        double* ptr = static_cast<double*>(buf.ptr);
        for (size_t i = 0; i < size; i++) {
            uint64_t idx = i;
            vsla_set_f64(tensor_, &idx, ptr[i]);
        }
    }
    
    ~PyVslaTensor() {
        if (tensor_) {
            vsla_free(tensor_);
        }
    }
    
    // Convert back to numpy array
    py::array_t<double> to_numpy() const {
        if (!tensor_) {
            throw std::runtime_error("Invalid tensor");
        }
        
        size_t size = tensor_->shape[0];
        auto result = py::array_t<double>(size);
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);
        
        for (size_t i = 0; i < size; i++) {
            uint64_t idx = i;
            ptr[i] = vsla_get_f64(tensor_, &idx);
        }
        
        return result;
    }
    
    // Addition operation
    PyVslaTensor add(const PyVslaTensor& other) const {
        if (!tensor_ || !other.tensor_) {
            throw std::runtime_error("Invalid tensor for addition");
        }
        
        size_t max_size = std::max(tensor_->shape[0], other.tensor_->shape[0]);
        vsla_tensor_t* result = vsla_new(1, &max_size, tensor_->model, VSLA_DTYPE_F64);
        
        if (!result) {
            throw std::runtime_error("Failed to create result tensor");
        }
        
        vsla_error_t err = vsla_add(result, tensor_, other.tensor_);
        if (err != VSLA_SUCCESS) {
            vsla_free(result);
            throw std::runtime_error("Addition failed: " + std::string(vsla_error_string(err)));
        }
        
        // Create wrapper for result
        PyVslaTensor result_wrapper(py::array_t<double>(1), tensor_->model);
        vsla_free(result_wrapper.tensor_);
        result_wrapper.tensor_ = result;
        
        return result_wrapper;
    }
    
    // Convolution operation (Model A only)
    PyVslaTensor convolve(const PyVslaTensor& other) const {
        if (!tensor_ || !other.tensor_) {
            throw std::runtime_error("Invalid tensor for convolution");
        }
        
        if (tensor_->model != VSLA_MODEL_A || other.tensor_->model != VSLA_MODEL_A) {
            throw std::runtime_error("Convolution only available in Model A");
        }
        
        size_t result_size = tensor_->shape[0] + other.tensor_->shape[0] - 1;
        vsla_tensor_t* result = vsla_new(1, &result_size, VSLA_MODEL_A, VSLA_DTYPE_F64);
        
        if (!result) {
            throw std::runtime_error("Failed to create result tensor");
        }
        
        vsla_error_t err = vsla_conv(result, tensor_, other.tensor_);
        if (err != VSLA_SUCCESS) {
            vsla_free(result);
            throw std::runtime_error("Convolution failed: " + std::string(vsla_error_string(err)));
        }
        
        // Create wrapper for result
        PyVslaTensor result_wrapper(py::array_t<double>(1), VSLA_MODEL_A);
        vsla_free(result_wrapper.tensor_);
        result_wrapper.tensor_ = result;
        
        return result_wrapper;
    }
    
    // Get shape
    std::vector<size_t> shape() const {
        if (!tensor_) {
            return {};
        }
        return std::vector<size_t>(tensor_->shape, tensor_->shape + tensor_->ndim);
    }
    
    // Get model
    std::string model() const {
        if (!tensor_) {
            return "unknown";
        }
        return tensor_->model == VSLA_MODEL_A ? "A" : "B";
    }

private:
    vsla_tensor_t* tensor_ = nullptr;
};

PYBIND11_MODULE(_core, m) {
    m.doc() = "VSLA: Variable-Shape Linear Algebra - Core C++ bindings";
    
    // Initialize VSLA library
    if (vsla_init() != VSLA_SUCCESS) {
        throw std::runtime_error("Failed to initialize VSLA library");
    }
    
    // Register cleanup function
    auto cleanup = []() {
        vsla_cleanup();
    };
    m.add_object("_cleanup", py::capsule(cleanup));
    
    // Expose model enum
    py::enum_<vsla_model_t>(m, "Model")
        .value("A", VSLA_MODEL_A, "Convolution semiring model")
        .value("B", VSLA_MODEL_B, "Kronecker semiring model");
    
    // Expose main tensor class
    py::class_<PyVslaTensor>(m, "Tensor")
        .def(py::init<py::array_t<double>, vsla_model_t>(), 
             py::arg("data"), py::arg("model") = VSLA_MODEL_A,
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
            return "VslaTensor(shape=" + shape_str + ", model=" + t.model() + ")";
        });
    
    // Utility functions
    m.def("add", [](const PyVslaTensor& a, const PyVslaTensor& b) {
        return a.add(b);
    }, "Add two tensors");
    
    m.def("convolve", [](const PyVslaTensor& a, const PyVslaTensor& b) {
        return a.convolve(b);
    }, "Convolve two tensors");
    
    // Version info
    m.attr("__version__") = "0.1.0";
}