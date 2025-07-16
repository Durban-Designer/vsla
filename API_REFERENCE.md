# VSLA Library API Reference

Complete reference for all public APIs in libvsla v1.0.0.

## 📋 Table of Contents

- [Error Handling](#error-handling)
- [Core Types](#core-types)
- [Library Management](#library-management)
- [Tensor Creation](#tensor-creation)
- [Data Access](#data-access)
- [Variable-Shape Operations](#variable-shape-operations)
- [Shape Manipulation](#shape-manipulation)
- [Utility Functions](#utility-functions)

## 🚨 Error Handling

### Error Codes

```c
typedef enum {
    VSLA_SUCCESS = 0,              // Operation completed successfully
    VSLA_ERROR_NULL_POINTER,       // Null pointer passed where not allowed
    VSLA_ERROR_INVALID_ARGUMENT,   // Invalid argument provided
    VSLA_ERROR_MEMORY,             // Memory allocation failed
    VSLA_ERROR_DIMENSION_MISMATCH, // Dimension mismatch in operation
    VSLA_ERROR_INVALID_MODEL,      // Invalid model specified
    VSLA_ERROR_INVALID_DTYPE,      // Invalid data type specified
    VSLA_ERROR_IO,                 // I/O operation failed
    VSLA_ERROR_NOT_IMPLEMENTED,    // Feature not yet implemented
    VSLA_ERROR_INVALID_RANK,       // Invalid rank (must be 0-255)
    VSLA_ERROR_OVERFLOW,           // Numeric overflow detected
    VSLA_ERROR_FFT,                // FFT operation failed
    VSLA_ERROR_INVALID_FILE,       // Invalid file format
    VSLA_ERROR_INCOMPATIBLE_MODELS // Incompatible models in operation
} vsla_error_t;
```

### Error Utilities

#### `vsla_error_string()`
```c
const char* vsla_error_string(vsla_error_t error);
```
**Purpose**: Get human-readable error message

**Parameters**:
- `error`: Error code to convert

**Returns**: String describing the error

**Example**:
```c
vsla_error_t err = vsla_set_f64(tensor, indices, value);
if (err != VSLA_SUCCESS) {
    printf("Error: %s\n", vsla_error_string(err));
}
```

## 🏗️ Core Types

### Models

```c
typedef enum {
    VSLA_MODEL_A = 0,  // Model A: Convolution-based (commutative)
    VSLA_MODEL_B = 1   // Model B: Kronecker product-based (non-commutative)
} vsla_model_t;
```

### Data Types

```c
typedef enum {
    VSLA_DTYPE_F64 = 0,  // 64-bit floating point (double)
    VSLA_DTYPE_F32 = 1   // 32-bit floating point (float)
} vsla_dtype_t;
```

### Tensor Structure

```c
typedef struct {
    uint8_t    rank;      // Number of axes (dimensions), 0-255
    uint8_t    model;     // Model: 0 = convolution, 1 = Kronecker
    uint8_t    dtype;     // Data type: 0 = f64, 1 = f32
    uint8_t    flags;     // Reserved for future use

    uint64_t  *shape;     // Logical extent per axis (length = rank)
    uint64_t  *cap;       // Padded/allocated extent per axis
    uint64_t  *stride;    // Byte strides for row-major traversal
    void      *data;      // Contiguous buffer, 64-byte aligned
} vsla_tensor_t;
```

**Important**: Never modify the tensor structure directly. Use the provided API functions.

## 🔧 Library Management

#### `vsla_init()`
```c
vsla_error_t vsla_init(void);
```
**Purpose**: Initialize the VSLA library

**Returns**: `VSLA_SUCCESS` or error code

**Details**: Optional but recommended. Sets up FFTW plans if available.

#### `vsla_cleanup()`
```c
vsla_error_t vsla_cleanup(void);
```
**Purpose**: Clean up library resources

**Returns**: `VSLA_SUCCESS` or error code

**Details**: Cleans up FFTW plans and other global resources.

#### `vsla_version()`
```c
const char* vsla_version(void);
```
**Purpose**: Get library version string

**Returns**: Version string (e.g., "1.0.0")

#### `vsla_has_fftw()`
```c
int vsla_has_fftw(void);
```
**Purpose**: Check if FFTW support is compiled in

**Returns**: 1 if FFTW available, 0 otherwise

## 🏗️ Tensor Creation

#### `vsla_new()`
```c
vsla_tensor_t* vsla_new(uint8_t rank, const uint64_t shape[], 
                        vsla_model_t model, vsla_dtype_t dtype);
```
**Purpose**: Create a new tensor

**Parameters**:
- `rank`: Number of dimensions (0-255)
- `shape`: Array of dimension sizes (length = rank)
- `model`: Model type (`VSLA_MODEL_A` or `VSLA_MODEL_B`)
- `dtype`: Data type (`VSLA_DTYPE_F64` or `VSLA_DTYPE_F32`)

**Returns**: New tensor or `NULL` on error

**Details**:
- Allocates 64-byte aligned memory
- Capacity set to next power-of-2 for each dimension
- Data initialized to zero

**Example**:
```c
uint64_t shape[] = {3, 4};
vsla_tensor_t* tensor = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
if (!tensor) {
    // Handle error
}
```

#### `vsla_zeros()`
```c
vsla_tensor_t* vsla_zeros(uint8_t rank, const uint64_t shape[],
                          vsla_model_t model, vsla_dtype_t dtype);
```
**Purpose**: Create tensor filled with zeros

**Parameters**: Same as `vsla_new()`

**Returns**: New zero tensor or `NULL` on error

#### `vsla_ones()`
```c
vsla_tensor_t* vsla_ones(uint8_t rank, const uint64_t shape[],
                         vsla_model_t model, vsla_dtype_t dtype);
```
**Purpose**: Create tensor filled with ones

**Parameters**: Same as `vsla_new()`

**Returns**: New tensor filled with ones or `NULL` on error

#### `vsla_zero_element()`
```c
vsla_tensor_t* vsla_zero_element(vsla_model_t model, vsla_dtype_t dtype);
```
**Purpose**: Create semiring zero element (empty tensor)

**Parameters**:
- `model`: Model type
- `dtype`: Data type

**Returns**: Zero element (rank-0 tensor) or `NULL` on error

#### `vsla_one_element()`
```c
vsla_tensor_t* vsla_one_element(vsla_model_t model, vsla_dtype_t dtype);
```
**Purpose**: Create semiring one element

**Parameters**:
- `model`: Model type
- `dtype`: Data type

**Returns**: One element (1D tensor with single element 1) or `NULL` on error

#### `vsla_copy()`
```c
vsla_tensor_t* vsla_copy(const vsla_tensor_t* tensor);
```
**Purpose**: Create deep copy of tensor

**Parameters**:
- `tensor`: Tensor to copy

**Returns**: New tensor with copied data or `NULL` on error

#### `vsla_free()`
```c
void vsla_free(vsla_tensor_t* tensor);
```
**Purpose**: Free tensor and all allocated memory

**Parameters**:
- `tensor`: Tensor to free (can be `NULL`)

**Details**: Safe to call with `NULL` pointer

## 📊 Data Access

#### `vsla_get_f64()`
```c
vsla_error_t vsla_get_f64(const vsla_tensor_t* tensor, const uint64_t indices[], 
                          double* value);
```
**Purpose**: Get element value with type conversion

**Parameters**:
- `tensor`: Input tensor
- `indices`: Array of indices (length = rank)
- `value`: Output value

**Returns**: `VSLA_SUCCESS` or error code

**Details**: Automatically converts from f32 to f64 if needed

**Example**:
```c
uint64_t indices[] = {1, 2};
double value;
vsla_error_t err = vsla_get_f64(tensor, indices, &value);
if (err == VSLA_SUCCESS) {
    printf("Value at [1,2]: %f\n", value);
}
```

#### `vsla_set_f64()`
```c
vsla_error_t vsla_set_f64(vsla_tensor_t* tensor, const uint64_t indices[], 
                          double value);
```
**Purpose**: Set element value with type conversion

**Parameters**:
- `tensor`: Tensor to modify
- `indices`: Array of indices (length = rank)
- `value`: Value to set

**Returns**: `VSLA_SUCCESS` or error code

**Details**: Automatically converts from f64 to f32 if needed

#### `vsla_fill()`
```c
vsla_error_t vsla_fill(vsla_tensor_t* tensor, double value);
```
**Purpose**: Fill all elements with a constant value

**Parameters**:
- `tensor`: Tensor to fill
- `value`: Value to fill with

**Returns**: `VSLA_SUCCESS` or error code

**Details**: Rejects NaN and infinity values

#### `vsla_get_ptr()`
```c
void* vsla_get_ptr(const vsla_tensor_t* tensor, const uint64_t indices[]);
```
**Purpose**: Get pointer to element (advanced usage)

**Parameters**:
- `tensor`: Input tensor
- `indices`: Array of indices

**Returns**: Pointer to element or `NULL` if out of bounds

**Warning**: Direct pointer access bypasses bounds checking. Use with caution.

## 🧮 Variable-Shape Operations

#### `vsla_add()`
```c
vsla_error_t vsla_add(vsla_tensor_t* out, const vsla_tensor_t* a, 
                      const vsla_tensor_t* b);
```
**Purpose**: Element-wise addition with automatic padding

**Parameters**:
- `out`: Output tensor (pre-allocated)
- `a`: First input tensor
- `b`: Second input tensor

**Returns**: `VSLA_SUCCESS` or error code

**Details**:
- Automatically pads to compatible shapes
- Output shape becomes max of input shapes
- Zero-padding applied where needed

**Example**:
```c
// Add [1,2,3] + [1,2,3,4,5] = [2,4,6,4,5]
uint64_t out_shape[] = {5}; // max(3, 5)
vsla_tensor_t* result = vsla_zeros(1, out_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
vsla_error_t err = vsla_add(result, tensor_a, tensor_b);
```

#### `vsla_sub()`
```c
vsla_error_t vsla_sub(vsla_tensor_t* out, const vsla_tensor_t* a, 
                      const vsla_tensor_t* b);
```
**Purpose**: Element-wise subtraction with automatic padding

**Parameters**: Same as `vsla_add()`

**Returns**: `VSLA_SUCCESS` or error code

#### `vsla_scale()`
```c
vsla_error_t vsla_scale(vsla_tensor_t* out, const vsla_tensor_t* tensor, 
                        double scalar);
```
**Purpose**: Scale tensor by scalar value

**Parameters**:
- `out`: Output tensor (can be same as input for in-place)
- `tensor`: Input tensor
- `scalar`: Scalar multiplier

**Returns**: `VSLA_SUCCESS` or error code

## 🔄 Shape Manipulation

#### `vsla_pad_rank()`
```c
vsla_error_t vsla_pad_rank(vsla_tensor_t* tensor, uint8_t new_rank, 
                           const uint64_t target_cap[]);
```
**Purpose**: Zero-copy rank expansion

**Parameters**:
- `tensor`: Tensor to expand
- `new_rank`: New rank (must be >= current rank)
- `target_cap`: Target capacities for new dimensions (can be `NULL`)

**Returns**: `VSLA_SUCCESS` or error code

**Details**:
- Zero-copy operation (no data movement)
- New dimensions have shape 0 (implicit zeros)
- Preserves existing data

**Example**:
```c
// Expand 2D tensor to 3D
vsla_error_t err = vsla_pad_rank(tensor, 3, NULL);
```

## 📐 Utility Functions

#### `vsla_numel()`
```c
uint64_t vsla_numel(const vsla_tensor_t* tensor);
```
**Purpose**: Get total number of elements (based on shape)

**Parameters**:
- `tensor`: Input tensor

**Returns**: Number of elements or 0 if tensor is `NULL`

#### `vsla_capacity()`
```c
uint64_t vsla_capacity(const vsla_tensor_t* tensor);
```
**Purpose**: Get total allocated capacity

**Parameters**:
- `tensor`: Input tensor

**Returns**: Total capacity or 0 if tensor is `NULL`

#### `vsla_shape_equal()`
```c
int vsla_shape_equal(const vsla_tensor_t* a, const vsla_tensor_t* b);
```
**Purpose**: Check if two tensors have the same shape

**Parameters**:
- `a`: First tensor
- `b`: Second tensor

**Returns**: 1 if shapes match, 0 otherwise

**Details**: Returns 0 if either tensor is `NULL`

#### `vsla_norm()`
```c
vsla_error_t vsla_norm(const vsla_tensor_t* tensor, double* norm);
```
**Purpose**: Compute Frobenius norm

**Parameters**:
- `tensor`: Input tensor
- `norm`: Output norm value

**Returns**: `VSLA_SUCCESS` or error code

**Formula**: `||x|| = sqrt(sum(x_i^2))`

#### `vsla_sum()`
```c
vsla_error_t vsla_sum(const vsla_tensor_t* tensor, double* sum);
```
**Purpose**: Compute sum of all elements

**Parameters**:
- `tensor`: Input tensor
- `sum`: Output sum value

**Returns**: `VSLA_SUCCESS` or error code

#### `vsla_print()`
```c
void vsla_print(const vsla_tensor_t* tensor, const char* name);
```
**Purpose**: Print tensor information for debugging

**Parameters**:
- `tensor`: Tensor to print
- `name`: Optional name for the tensor (can be `NULL`)

**Output**: Prints rank, model, dtype, shape, capacity, and sample data

## 💡 Usage Patterns

### Basic Workflow
```c
// 1. Initialize library (optional)
vsla_init();

// 2. Create tensors
vsla_tensor_t* a = vsla_new(2, (uint64_t[]){3, 4}, VSLA_MODEL_A, VSLA_DTYPE_F64);
vsla_tensor_t* b = vsla_ones(2, (uint64_t[]){5, 2}, VSLA_MODEL_A, VSLA_DTYPE_F64);

// 3. Fill with data
vsla_fill(a, 2.0);

// 4. Perform operations
uint64_t out_shape[] = {5, 4}; // max shapes
vsla_tensor_t* result = vsla_zeros(2, out_shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
vsla_add(result, a, b);

// 5. Access results
double value;
vsla_get_f64(result, (uint64_t[]){0, 0}, &value);

// 6. Clean up
vsla_free(a);
vsla_free(b);
vsla_free(result);
vsla_cleanup();
```

### Error Handling Pattern
```c
vsla_error_t err;
vsla_tensor_t* tensor = vsla_new(2, shape, VSLA_MODEL_A, VSLA_DTYPE_F64);
if (!tensor) {
    fprintf(stderr, "Failed to create tensor\n");
    return -1;
}

err = vsla_fill(tensor, 1.0);
if (err != VSLA_SUCCESS) {
    fprintf(stderr, "Fill failed: %s\n", vsla_error_string(err));
    vsla_free(tensor);
    return -1;
}

vsla_free(tensor);
```

### Type Safety Pattern
```c
// Works with both f32 and f64 tensors
vsla_tensor_t* f32_tensor = vsla_new(1, (uint64_t[]){10}, VSLA_MODEL_A, VSLA_DTYPE_F32);
vsla_tensor_t* f64_tensor = vsla_new(1, (uint64_t[]){10}, VSLA_MODEL_A, VSLA_DTYPE_F64);

// API automatically handles type conversion
double value = 3.14159;
vsla_set_f64(f32_tensor, (uint64_t[]){0}, value); // Converts to f32
vsla_set_f64(f64_tensor, (uint64_t[]){0}, value); // Stores as f64

double retrieved;
vsla_get_f64(f32_tensor, (uint64_t[]){0}, &retrieved); // Converts from f32
vsla_get_f64(f64_tensor, (uint64_t[]){0}, &retrieved); // Direct f64 access
```

## ⚠️ Important Notes

### Memory Management
- Always call `vsla_free()` for every tensor created
- `vsla_free()` is safe to call with `NULL`
- Don't modify tensor structure fields directly
- Use provided API functions for all operations

### Thread Safety
- Individual tensors are not thread-safe for modification
- Multiple threads can safely read from the same tensor
- Library initialization (`vsla_init()`) should be called once

### Performance Tips
- Pre-allocate result tensors when possible
- Use power-of-2 dimensions for optimal memory usage
- Prefer f32 for memory-constrained applications
- Call `vsla_init()` for FFTW optimization (future)

### Limitations
- Maximum rank: 255 dimensions
- Maximum dimension size: Limited by available memory
- Tensor size limit: 1TB per dimension
- No sparse tensor support (yet)

---

This API reference covers all currently implemented functions in libvsla v1.0.0. For examples and advanced usage, see the `examples/` directory and `README.md`.