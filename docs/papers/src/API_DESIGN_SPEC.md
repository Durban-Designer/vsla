# VSLA API Design Specification

This document contains the detailed API specifications that were removed from the academic paper to maintain focus on theoretical contributions.

## Variable-Shape Tensor Operations

### Core Transform Operations

#### Variable-Shape Slicing and Indexing
**Operations:** `VS_SLICE`, `VS_INDEX`

These operations allow for extracting sub-tensors or specific elements based on their logical dimensions, rather than fixed-size bounding boxes. By directly operating on the minimal representative, `VS_SLICE` efficiently retrieves only the materialized non-zero data corresponding to the requested sub-region, avoiding overhead from implicit padding.

**Proposed API:**
```c
vsla_tensor_t* vsla_slice(const vsla_tensor_t* tensor, 
                         const vsla_slice_spec_t* slice_spec);
vsla_scalar_t vsla_index(const vsla_tensor_t* tensor, 
                        const int* indices, int ndim);
```

**Applications:** Vital for isolating specific spatial or temporal domains in simulations.

#### Variable-Shape Permutation and Transposition
**Operations:** `VS_PERMUTE`, `VS_TRANSPOSE`

Standard tensor permutation, which reorders axes, made VSLA-aware. For sparse representations, such operations primarily involve rearranging metadata (e.g., stride information, dimension order) rather than extensive data movement of padded elements.

**Proposed API:**
```c
vsla_tensor_t* vsla_permute(const vsla_tensor_t* tensor, 
                           const int* perm_order, int ndim);
vsla_tensor_t* vsla_transpose(const vsla_tensor_t* tensor, 
                             int axis1, int axis2);
```

**Applications:** Essential for aligning simulation data for different computational kernels.

#### Variable-Shape Reshaping
**Operations:** `VS_RESHAPE`

This operation allows changing the logical shape of a VSLA tensor without altering its underlying data elements, provided the total number of elements remains consistent. Due to VSLA's sparse-by-design memory model, reshaping operations are largely metadata transformations.

**Proposed API:**
```c
vsla_tensor_t* vsla_reshape(const vsla_tensor_t* tensor, 
                           const int* new_shape, int ndim);
```

**Applications:** Beneficial for converting between different conceptual representations of simulation data.

#### Sparse-Aware Scatter/Gather Operations
**Operations:** `VS_SCATTER`, `VS_GATHER`

These operations are fundamental for non-contiguous data movement. `VS_SCATTER` writes elements from a source tensor to specific (potentially sparse) locations in a target VSLA tensor, while `VS_GATHER` collects elements from specific locations.

**Proposed API:**
```c
void vsla_scatter(vsla_tensor_t* target, const vsla_tensor_t* source, 
                  const vsla_tensor_t* indices);
vsla_tensor_t* vsla_gather(const vsla_tensor_t* source, 
                          const vsla_tensor_t* indices);
```

**Applications:** Crucial for particle-in-cell methods, dynamic mesh updates, or any simulation where data elements move or interact non-locally.

## Implementation Considerations

### Performance Characteristics
- All operations designed to work only with materialized non-zero elements
- Metadata transformations preferred over data movement where possible
- Complexity targets: O(nnz) for most operations where nnz = number of non-zero elements

### Memory Model Integration
- Operations must preserve VSLA equivalence class membership
- Automatic shape promotion when necessary
- Sparse storage maintained throughout all transformations

### Error Handling
- Shape compatibility checking
- Index bounds validation on logical dimensions
- Memory allocation failure handling

## Future API Extensions

### Advanced Operations
- Sparse convolution with custom kernels
- Multi-tensor operations (batch processing)
- Streaming operations for real-time data

### Integration Points
- NumPy compatibility layer
- JAX/PyTorch custom operators
- GPU acceleration interfaces

This specification serves as a roadmap for implementing the complete VSLA transformation suite while maintaining the mathematical rigor established in the theoretical framework.