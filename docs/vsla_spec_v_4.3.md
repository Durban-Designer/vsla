# VSLA Implementation Guide v4.3 — Programs, Pipelines, Arenas & Autograd

**Objective**: Provide a mathematically rigorous and implementation-ready specification for the VSLA C backend. This document is normative: coding agents must adhere to it unless an explicit deviation note accompanies the change.

## v4.3 Changes Summary:

*   **Formalized IR and Program Lifecycle**: Fully integrates the `vsla_program_t`, `vsla_plan_t`, and `vsla_backend_t` concepts from v4.1/4.2, detailing their construction, planning, and execution.
*   **Explicit Memory Planning**: Comprehensive specification of liveness analysis, linear-scan coloring, and arena management to ensure zero allocations in the execution hot path.
*   **Expanded Autograd**: Provides a more complete and detailed Vector-Jacobian Product (VJP) table for a wider range of operations, including specifics for handling broadcasting and reductions.
*   **Consolidated FFT Details**: Integrates the detailed FFT implementation specification from v3.2 into the v4.3 guide.
*   **Clarity on `vsla_shrink`**: Emphasizes the deferred nature of `vsla_shrink` to prevent quadratic performance cascades.
*   **Cross-Program Composition**: Defines mechanisms for linking programs and managing shared memory.
*   **Refined Error Handling**: Reinforces `vsla_error_t` usage and error state management.
*   **Updated Testing Strategy**: Consolidates and expands testing requirements, including autograd checks.
*   **Comprehensive Terminology**: Ensures consistent use of a precise glossary throughout.
*   **Migration Path**: Provides a clear guide for transitioning from v3.2.

## Table of Contents
1.  [Core Concepts (precise glossary)](#1-core-concepts-precise-glossary)
2.  [High-Level Architecture](#2-high-level-architecture)
3.  [IR (Intermediate Representation)](#3-ir-intermediate-representation)
    *   [3.1. Data structures](#31-data-structures)
    *   [3.2. Validation Checklist](#32-validation-checklist)
4.  [Shape / Capacity Inference](#4-shape--capacity-inference)
    *   [4.1. Rules](#41-rules)
    *   [4.2. FFT Workspace Sizing](#42-fft-workspace-sizing)
    *   [4.3. Higher-Rank Tensor Conventions](#43-higher-rank-tensor-conventions)
5.  [Memory Planning](#5-memory-planning)
    *   [5.1. Data Structures](#51-data-structures)
    *   [5.2. Liveness Computation](#52-liveness-computation)
    *   [5.3. Linear-Scan Buffer Coloring](#53-linear-scan-buffer-coloring)
    *   [5.4. WORKSPACE Special Rule](#54-workspace-special-rule)
    *   [5.5. Output of Planning](#55-output-of-planning)
6.  [Arenas](#6-arenas)
    *   [6.1. Arena Classes](#61-arena-classes)
    *   [6.2. Alignment Policy](#62-alignment-policy)
    *   [6.3. Profiles (Bucketing)](#63-profiles-bucketing)
    *   [6.4. vsla_shrink Policy](#64-vsla_shrink-policy)
    *   [6.5. vsla_shrink Policy and Minimal Representatives](#65-vsla_shrink-policy-and-minimal-representatives)
7.  [Execution (CPU Reference)](#7-execution-cpu-reference)
    *   [7.1. Segment Representation](#71-segment-representation)
    *   [7.2. CPU execute() Outline](#72-cpu-execute-outline)
    *   [7.3. Fusion Rules](#73-fusion-rules)
    *   [7.4. Streams](#74-streams)
8.  [Parallelism and Hardware Acceleration](#8-parallelism-and-hardware-acceleration)
    *   [8.1. CPU Multithreading](#81-cpu-multithreading)
    *   [8.2. GPU Acceleration (CUDA/ROCm)](#82-gpu-acceleration-cudarocm)
9.  [Autograd (Reverse-Mode AD)](#9-autograd-reverse-mode-ad)
    *   [9.1. Build Backward Program Pᵀ](#91-build-backward-program-p)
    *   [9.2. Checkpointing](#92-checkpointing)
    *   [9.3. Planning & Execution for Backward Pass](#93-planning--execution-for-backward-pass)
    *   [9.4. Vector-Jacobian Product (VJP) Table](#94-vector-jacobian-product-vjp-table)
10. [Cross-Program / Cross-Arena Composition](#10-cross-program--cross-arena-composition)
    *   [10.1. Import/Export Values (vsla_link_t)](#101-importexport-values-vsla_link_t)
    *   [10.2. Shared Arena Manager](#102-shared-arena-manager)
11. [Backend Abstraction (C vtable)](#11-backend-abstraction-c-vtable)
12. [Public C API (End-to-End Lifecycle)](#12-public-c-api-end-to-end-lifecycle)
    *   [12.1. Compile (Forward)](#121-compile-forward)
    *   [12.2. Execute (Forward)](#122-execute-forward)
    *   [12.3. Compile & Run (Backward)](#123-compile--run-backward)
    *   [12.4. Profiles Usage](#124-profiles-usage)
13. [Error Handling and Reporting](#13-error-handling-and-reporting)
    *   [13.1. Error Codes (`vsla_error_t`)](#131-error-codes-vsla_error_t)
    *   [13.2. Retrieving Error Information](#132-retrieving-error-information)
    *   [13.3. Best Practices](#133-best-practices)
14. [Canonical Applications and Examples](#14-canonical-applications-and-examples)
    *   [14.1. Transformer Worked Example](#141-transformer-worked-example)
    *   [14.2. Tensor Pyramid Construction](#142-tensor-pyramid-construction)
15. [Numerics & Precision Policy](#15-numerics--precision-policy)
16. [Testing Matrix](#16-testing-matrix)
17. [Performance Checklist](#17-performance-checklist)
18. [Migration Guide (v3.2 → v4.3)](#18-migration-guide-v32--v43)
19. [Future Work](#19-future-work)

## 1. Core Concepts (precise glossary)
*   **IR (Intermediate Representation)**: SSA-like directed acyclic graph (DAG) of Nodes (operations) producing Values (tensors). Topologically sortable.
*   **Program**: A frozen, validated IR along with metadata such as constants, parameters, and input/output signatures.
*   **Plan**: The deterministic execution schedule and memory mapping for a Program, which includes: value liveness information, colored buffers, arena sizes, FFT plan cache, and a list of fused segments.
*   **Arena**: A single contiguous block of memory designated for one class of tensors (e.g., `PARAMS`, `ACTIVATIONS`, `WORKSPACE`, `GRADS`). Values point to fixed offsets within their assigned arena.
*   **Executor**: The component that runs a Program according to its Plan on a specific Backend.
*   **Backend**: A swappable device implementation (e.g., CPU, CUDA, HIP/ROCm, Metal, Vulkan, OpenCL) exposed via a C vtable.
*   **Profile (Bucket)**: A pre-compiled set consisting of a Program, its Plan, and an Executor, optimized for a specific shape family (e.g., `seq_len = 1024`). This allows for repeated runs without reallocation.
*   **Tape**: An autograd side-structure that captures the subset of forward values necessary for computing the backward pass.
*   **Checkpoint**: A forward node explicitly marked to store its outputs; other intermediate values may be recomputed during the backward pass if not checkpointed.
*   **Minimal Representative**: Each equivalence class of variable-shape vectors has a unique representative whose shape has no trailing zero hyperplanes. Runtime storage must always use minimal representatives unless explicitly stated (e.g., before `vsla_shrink` or serialization).
*   **Ambient Promotion**: The process where operands are implicitly coerced to a common dimension (the element-wise maximum of their logical dimensions) before computation, while preserving sparsity and algebraic properties. This avoids materializing padded zeros.
*   **Capacity**: The physical memory allocated for a tensor, which is typically `next_pow2(shape[i])` for each dimension `i`, and always `>= shape[i]`.
*   **Modal verbs (RFC 2119 style)**:
    *   **MUST** — mandatory for correctness.
    *   **SHOULD** — strong recommendation for performance or safety.
    *   **MAY** — optional/implementation detail.

## 2. High-Level Architecture
The VSLA system follows a distinct compilation and execution pipeline to ensure high performance and efficient resource management.

User Graph → Build IR → Infer Shapes/Capacity → Validate → Plan Memory (Liveness + Coloring) → Produce Arenas & Offsets → Lower to Executor (CPU ref) / Record CUDA Graph → Execute
↘
Build Backward IR (Autograd) → Plan → Execute

**Key Invariants**:
*   **MUST NOT** allocate memory during the `execute()` hot path.
*   Plans **MUST** be reusable without modification for every batch within the same profile.
*   Programs **MUST** be immutable after planning; rebuild the program to change shapes or operations.

## 3. IR (Intermediate Representation)
The IR provides a formal, machine-readable representation of the computation graph.

### 3.1. Data structures
```c
typedef enum {
    VSLA_TENSOR_PARAM,   // Parameter (weight/bias)
    VSLA_TENSOR_INPUT,   // Program input
    VSLA_TENSOR_OUTPUT,  // Program output
    VSLA_TENSOR_TEMP,    // Internal intermediate value
} vsla_valuetag_t; [cite: 4, 5]

typedef enum {
    VSLA_ARENA_PARAMS = 0,    // Parameters, typically read-only during forward pass
    VSLA_ARENA_ACTIVATIONS = 1, // Intermediate activations (retained for backward if not checkpointed)
    VSLA_ARENA_WORKSPACE = 2, // Scratch memory for transient computations (e.g., FFT buffers)
    VSLA_ARENA_GRADS = 3,     // Gradients w.r.t. parameters and activations
    VSLA_ARENA_EXTERNAL = 4,  // Memory managed externally, linked via bindings
} vsla_arenaclass_t; [cite: 4, 5]

typedef struct {
    uint32_t        id;            // SSA id: Unique identifier for the value within the program
    vsla_dtype_t    dtype;         // Data type of the tensor elements (e.g., float32, float64)
    uint8_t         rank;          // Number of axes (dimensions) of the tensor
    uint64_t*       shape;         // Minimal logical dimensions of the tensor (length == rank)
    uint64_t*       capacity;      // Physical allocation dimensions, typically next_pow2(shape[i]) (length == rank)
    vsla_valuetag_t tag;           // Classification of the value (parameter, input, output, temp)
    vsla_arenaclass_t arena_class; // The memory arena this value belongs to
    uint64_t        arena_offset;  // Byte offset within its assigned arena (filled by planner)
} vsla_value_t; [cite: 4, 5]

// --- Typed Operation Attributes ---
typedef struct { int axis; } vsla_reduce_attrs_t;
typedef struct { double value; } vsla_scalar_attrs_t;
typedef struct { int* permutation; } vsla_transpose_attrs_t;
typedef struct { uint64_t* shape; int rank; } vsla_reshape_attrs_t;
typedef struct { int window_size; int stride; } vsla_window_attrs_t;

// Ops (extendable) — all are *pure* functions of their inputs
// (side effects exist only for PARAM writes during bind/load).
typedef enum {
    VSLA_OP_ADD, VSLA_OP_SUB, VSLA_OP_HADAMARD, [cite: 4, 5]
    VSLA_OP_CONV1D, VSLA_OP_KRONECKER, [cite: 4, 5]
    VSLA_OP_SOFTMAX, VSLA_OP_LAYERNORM, VSLA_OP_RMSNORM, [cite: 4]
    VSLA_OP_MATMUL, [cite: 4]
    VSLA_OP_RELU, VSLA_OP_GELU, VSLA_OP_SIGMOID, VSLA_OP_TANH, [cite: 4]
    VSLA_OP_DROPOUT, [cite: 4]
    VSLA_OP_SUM, VSLA_OP_MEAN, [cite: 4]
    VSLA_OP_STACK, VSLA_OP_WINDOW, [cite: 4]
    VSLA_OP_RESHAPE, VSLA_OP_TRANSPOSE, VSLA_OP_SLICE, VSLA_OP_GATHER, VSLA_OP_SCATTER, [cite: 4]
    VSLA_OP_MASK, VSLA_OP_EXP, VSLA_OP_LOG, VSLA_OP_MUL_SCALAR, [cite: 4]
    VSLA_OP_PARAMETER, VSLA_OP_INPUT, VSLA_OP_OUTPUT, [cite: 4, 5]
} vsla_opkind_t; [cite: 4, 5]

typedef struct vsla_node_s {
    uint32_t        id;          // Unique identifier for the node within the program
    vsla_opkind_t   kind;        // Type of operation this node performs
    vsla_model_t    model;       // VSLA_MODEL_A or VSLA_MODEL_B for semiring ops; ignore if N/A
    uint32_t*       inputs;      // Array of value IDs that are inputs to this node
    uint32_t        ninputs;     // Number of inputs
    uint32_t*       outputs;     // Array of value IDs that are outputs from this node
    uint32_t        noutputs;    // Number of outputs
    union {                      // Type-safe, discriminated union for attributes
        vsla_reduce_attrs_t reduce;
        vsla_scalar_attrs_t scalar;
        vsla_transpose_attrs_t transpose;
        vsla_reshape_attrs_t reshape;
        vsla_window_attrs_t window;
    } attrs;
} vsla_node_t; [cite: 4, 5]

typedef struct {
    vsla_node_t*  nodes;            // Array of nodes in topological order
    size_t        nnodes;           // Number of nodes
    vsla_value_t* values;           // Array of all values (tensors) in the program
    size_t        nvalues;          // Number of values
    uint32_t*     program_inputs;   // Array of value IDs that are external inputs to the program
    size_t        n_program_inputs; // Number of program inputs
    uint32_t*     program_outputs;  // Array of value IDs that are external outputs of the program
    size_t        n_program_outputs; // Number of program outputs
} vsla_program_t; [cite: 4, 5]
```

### 3.2. Validation Checklist (MUST)
Before planning or execution, a `vsla_program_t` **MUST** pass these validation checks:

*   The graph **MUST** be acyclic (a topological order **MUST** exist).
*   All node inputs and outputs **MUST** refer to existing `vsla_value_t` IDs.
*   Data types (`dtype`) **MUST** match per the operation's rules (e.g., all inputs to an `ADD` operation **MUST** have compatible dtypes).
*   The `vsla_model_t` (A or B) **MUST** be consistent where required for semiring operations.
*   No rank-0 tensors **MUST** be visible at the API boundary (internal handling may allow for rank==0 but public API should avoid it).
*   All shapes **MUST** be computed and non-negative.
*   `capacity[i]` **MUST** be `>= shape[i]` for all dimensions `i`.
*   All internal shape multiplications **MUST** pass overflow guards to prevent `UINT64_MAX` overflow.

## 4. Shape / Capacity Inference
Shape inference determines the logical dimensions of output tensors based on input shapes. Capacity inference determines the physical allocation size for each dimension.

### 4.1. Rules
Each operation defines its output shape purely as a function of its input shapes.

The logical region of a tensor, from index 0 up to `shape[i]-1` for each dimension, **MUST** be fully initialized.

The slack region (indices `>= shape[i]` but `< capacity[i]`) **MAY** contain uninitialized data and **MUST NOT** be read.

```c
for (node in topo_sorted(program.nodes)) {
    // Assume 'out' is the primary output value of the node, and 'a', 'b' are its primary inputs.
    // Full rules for all ops will be implemented.
    switch(node.kind) {
        case VSLA_OP_ADD: case VSLA_OP_SUB: case VSLA_OP_HADAMARD:
            // Element-wise operations: output shape is the element-wise maximum of input shapes (ambient shape).
            out.shape[i] = max(a.shape[i], b.shape[i]);
            break;
        case VSLA_OP_CONV1D:
            // Convolution (Model A): For vectors of length m and n, output length is m+n-1.
            out.shape[0] = (m == 0 || n == 0) ? 0 : (m + n - 1); [cite: 3]
            break;
        case VSLA_OP_KRONECKER:
            // Kronecker Product (Model B): For vectors of length m and n, output length is m*n.
            out.shape[0] = (m == 0 || n == 0) ? 0 : (m * n); [cite: 3]
            break;
        case VSLA_OP_STACK:
            // Stacking operator: rank increases by 1. New leading dimension is k (number of stacked tensors).
            // Remaining dimensions are the ambient shape of inputs.
            out.rank = input[0].rank + 1;
            out.shape[0] = k; // Number of stacked tensors
            for (i = 0; i < input[0].rank; ++i) {
                out.shape[i+1] = max_i(input[j].shape[i]); // element-wise max over all inputs for other dimensions
            }
            break;
        // ... (rules for other operations like Softmax, Matmul, Reshape, etc.)
    }
    // Capacity policy: For each dimension, capacity is the smallest power of two greater than or equal to the shape.
    // The planner MAY override this to shape[i] for small or static dimensions if it leads to better packing.
    for i in axes: value.capacity[i] = next_pow2(value.shape[i]); [cite: 5]
}
```

### 4.2. FFT Workspace Sizing
For convolution operations (`VSLA_OP_CONV1D`), FFTs require specific buffer lengths.

*   For vector-vector convolution (Model A), the FFT length `L` is `next_pow2(m + n - 1)`.
*   For higher-rank tensor convolution (iterating over element pairs), a single `L` is computed as `next_pow2(2 * d_max - 1)` where `d_max` is the maximum vdim across all elements (`T_ij`) involved in the matrix operation, ensuring enough room for the full convolution output.
*   FFT plans for a given `L` **SHOULD** be memoized (cached) to avoid redundant plan creation.

### 4.3. Higher-Rank Tensor Conventions
To ensure unambiguous behavior, operations on tensors with rank greater than 2 **MUST** follow these conventions:
*   **Element-wise Operations**: Apply to all elements. Shapes **MUST** be broadcastable to a common ambient shape.
*   **Reductions (`SUM`, `MEAN`)**: These operations take an `axis` attribute. The reduction is performed along the specified axis, and the rank of the output tensor is reduced by 1.
*   **`MATMUL`**: Operates on the last two dimensions of the input tensors. All preceding dimensions are treated as batch dimensions and **MUST** be broadcastable.
*   **`CONV1D`**: Operates on the last dimension of the input tensor. All preceding dimensions are treated as batch dimensions. `Conv2D` and `Conv3D` would operate on the last 2 and 3 dimensions, respectively.

## 5. Memory Planning
Memory planning is a crucial phase that ensures efficient memory utilization and the "no allocation in hot path" invariant. It involves determining the lifetime of each value and assigning it an offset within a pre-allocated arena.

### 5.1. Data Structures
```c
typedef struct {
    uint32_t value_id;     // ID of the vsla_value_t
    uint32_t start;        // First node index (inclusive) where the value is live
    uint32_t end;          // Last node index (inclusive) where the value is live
    size_t   bytes;        // Size of the value in bytes (product(capacity) * sizeof(dtype))
    vsla_arenaclass_t arena_class; // Which arena class this value belongs to
} vsla_interval_t; [cite: 4, 5]

typedef struct {
    uint64_t offset;       // Byte offset in arena
    uint64_t size;         // Length of this memory block
} vsla_block_t; [cite: 4]

// Represents the complete memory layout and execution schedule for a program.
typedef struct {
    uint64_t          arena_size[5]; // Total size in bytes for each arena class (PARAMS, ACTIVATIONS, WORKSPACE, GRADS, EXTERNAL)
    uint64_t*         offsets;       // Array parallel to program->values, storing byte offset within its arena
    vsla_schedule_t   schedule;      // Fused segments and their execution order (see Section 7.1)
    // Additional caches and metadata, e.g., FFT plan cache.
} vsla_plan_t; [cite: 4, 5]
```

### 5.2. Liveness Computation
Liveness analysis determines the lifetime (first and last use) of each value in the topologically sorted program.

```c
for (v in program->values) {
    v.first_use = UINT32_MAX; // Initialize to maximum possible value
    v.last_use  = 0;          // Initialize to minimum possible value
}

for (idx = 0; idx < program->nnodes; ++idx) {
    vsla_node_t* node = &program->nodes[idx];
    for (i = 0; i < node->noutputs; ++i) {
        uint32_t out_id = node->outputs[i];
        program->values[out_id].first_use = min(program->values[out_id].first_use, idx);
        program->values[out_id].last_use  = max(program->values[out_id].last_use, idx);
    }
    for (i = 0; i < node->ninputs; ++i) {
        uint32_t in_id = node->inputs[i];
        program->values[in_id].last_use   = max(program->values[in_id].last_use, idx);
    }
} [cite: 4, 5]
```
**Note**: `first_use` for program inputs (e.g., `VSLA_TENSOR_INPUT`, `VSLA_TENSOR_PARAM`) should be 0.

### 5.3. Linear-Scan Buffer Coloring
This algorithm assigns contiguous memory blocks (offsets) to values within their respective arena classes, minimizing total memory footprint by reusing space from dead values.

```c
for (arena_class in {VSLA_ARENA_PARAMS, VSLA_ARENA_ACTIVATIONS, VSLA_ARENA_WORKSPACE, VSLA_ARENA_GRADS}) { [cite: 4, 5]
    intervals = collect_intervals(program->values, arena_class); // Build interval_t for relevant values
    sort(intervals by start_node_idx asc, end_node_idx asc); // Sort intervals by start time, then end time

    free_list = min_heap_of_blocks(); // Store available free blocks, sorted by offset
    arena_tail = 0; // Current end of the arena, growing as new blocks are allocated

    for (it in intervals) {
        // 1) Expire dead blocks: Remove blocks from free_list whose end_node_idx is before it.start_node_idx
        expire_blocks_before(free_list, it.start);

        // 2) First-fit allocation: Try to find a suitable block in the free_list
        offset = find_first_fit(free_list, it.bytes, alignment=64); [cite: 4, 5]
        if (offset == INVALID) {
            // No fit found, extend the arena from the tail
            offset = align_up(arena_tail, 64); // Ensure 64-byte alignment
            arena_tail = offset + it.bytes;
        } else {
            // Fit found, adjust or remove the free block from the list
            adjust_free_list_entry(free_list, offset, it.bytes);
        }
        plan->offsets[it.value_id] = offset; // Assign the determined offset to the value
    }
    plan->arena_size[arena_class] = arena_tail; // Record the total size needed for this arena
}
```

### 5.4. WORKSPACE Special Rule
The `WORKSPACE` arena is distinct:

*   Its total size is computed as the peak sum of `bytes` for all `WORKSPACE` values that are simultaneously live.
*   Individual `WORKSPACE` values **MAY** be treated as a stack allocator during the linear-scan, meaning their specific offsets are transient during planning. However, for implementation simplicity and to align with the "no allocation in `execute()`" rule, it's often more pragmatic to assign each workspace-using operation its own static slice of the `WORKSPACE` arena, derived from the same linear-scan result.
*   The `plan->arena_size[WORKSPACE]` **MUST** be set to this computed maximum peak.

### 5.5. Output of Planning
Upon successful planning, `vsla_plan_t` will contain:

*   `plan->arena_size[arena_class]` for all arena classes, indicating the total bytes needed.
*   `value[i].arena_offset` for every value `i` in the program, specifying its byte offset within its assigned arena.
*   `plan->schedule`, a list of `vsla_segment_t` objects, defining the execution order and fused kernel groups.

## 6. Arenas
Arenas are pre-allocated, contiguous memory blocks that house tensors during execution. This model prevents runtime allocations in the hot path.

### 6.1. Arena Classes
*   **`VSLA_ARENA_PARAMS`**: For model weights, biases, and other static parameters. These are typically read-only during the forward pass and updated by optimizers after the backward pass.
*   **`VSLA_ARENA_ACTIVATIONS`**: For intermediate tensor outputs from forward pass operations. These may be retained for the backward pass (if not checkpointed).
*   **`VSLA_ARENA_WORKSPACE`**: For transient scratch memory needed by operations (e.g., FFT buffers, temporary sums). Memory in this arena is quickly reused.
*   **`VSLA_ARENA_GRADS`**: For storing gradients with respect to parameters and activations during the backward pass.
*   **`VSLA_ARENA_EXTERNAL`**: For memory buffers managed outside the VSLA system (e.g., user input/output buffers). These are bound to the program at execution time.

### 6.2. Alignment Policy
All memory offsets within an arena **MUST** be aligned to at least 64 bytes. This ensures optimal performance for SIMD operations and cache line utilization.

### 6.3. Profiles (Bucketing)
To avoid dynamic re-planning and re-allocation for variable-shape inputs, VSLA uses profiles:

*   A `vsla_profile_t` represents a pre-compiled (Program, Plan, Executor) tuple for a specific shape family (e.g., `batch_size=32`, `sequence_length=1024`).
*   Programs **MUST** be compiled and planned once per profile. At runtime, the input batch is routed to the appropriate pre-computed profile, enabling execution without any further memory allocations or re-planning.

### 6.4. vsla_shrink Policy
*   Kernels **MUST NOT** automatically perform `vsla_shrink` to their minimal representative. This prevents quadratic complexity cascades (`O(product(shape)*rank)`) that would arise from repeated shrinking operations.
*   `vsla_shrink()` **MAY** be run offline (e.g., during serialization) or explicitly invoked by the caller to convert a tensor to its minimal representative, primarily for storage efficiency.

### 6.5. `vsla_shrink` Policy and Minimal Representatives
The concept of a minimal representative is central to VSLA's efficiency, but its misuse can lead to performance bottlenecks. This section provides clear guidelines on how to handle `vsla_shrink`.

*   **What `vsla_shrink` Does**: The `vsla_shrink` operation reallocates a tensor to its minimal representative, removing any trailing zero hyperplanes. This is useful for serialization or when memory needs to be reclaimed, but it is a potentially expensive operation.
*   **The Dangers of Automatic Shrinking**: If `vsla_shrink` is performed automatically after every operation, it can lead to a quadratic complexity cascade. For example, if you are building up a tensor by repeatedly appending a small chunk, each append operation would trigger a full reallocation and copy of the entire tensor. This would be highly inefficient.
*   **When to Use `vsla_shrink`**: `vsla_shrink` should be used sparingly and only when you are sure that the memory savings are worth the cost of the reallocation. Good candidates for `vsla_shrink` are tensors that have a large number of trailing zeros and are not expected to grow any further.
*   **When to Avoid `vsla_shrink`**: Avoid using `vsla_shrink` in hot loops or on tensors that are frequently modified. In these cases, the cost of the repeated reallocations will likely outweigh any memory savings.
*   **Serialization**: `vsla_shrink` is particularly useful when serializing a tensor. By shrinking the tensor to its minimal representative before serialization, you can significantly reduce the amount of data that needs to be written to disk.

## 7. Execution (CPU Reference)
The Executor is responsible for running the program on the target backend according to the plan. This section outlines the CPU reference implementation, which **SHOULD** be multithreaded.

### 7.1. Segment Representation
The execution plan (`vsla_schedule_t` within `vsla_plan_t`) consists of a sequence of segments. Each segment represents a fused computational block.

```c
typedef vsla_error_t (*vsla_segment_fn)(const vsla_plan_t* plan,
                                      const vsla_value_t* values,
                                      const vsla_node_t* nodes,
                                      size_t first_node, size_t last_node, // Inclusive range of nodes for this segment
                                      const vsla_bindings_t* bind,       // Arena pointers
                                      vsla_stream_t* stream);            // Opaque stream handle for device/threading

typedef struct {
    size_t first_node, last_node; // Inclusive range of IR nodes covered by this segment
    vsla_segment_fn fn;           // Pointer to the specialized (fused) kernel function
} vsla_segment_t; [cite: 4, 5]

typedef struct {
    vsla_segment_t* segments;
    size_t          nsegments;
} vsla_schedule_t; // Embedded inside vsla_plan_t
```

### 7.2. CPU `execute()` Outline
```c
vsla_error_t vsla_program_execute(const vsla_program_t* P,
                                const vsla_plan_t* plan,
                                const vsla_bindings_t* bind, // Contains pointers to pre-allocated arenas
                                const vsla_io_t* in,         // Pointers to program input data (external)
                                vsla_io_t* out,              // Pointers to program output data (external)
                                vsla_stream_t* stream) {     // Opaque stream handle

    // 0) Map input/output host pointers into arena views if needed (for VSLA_ARENA_EXTERNAL)
    // This step ensures that the 'bind' structure correctly points to the external memory for program I/O.
    // No allocation should occur here.
    bind_program_ios(P, plan, bind, in, out);

    // 1) Iterate through each fused segment in the plan's schedule
    for (s = 0; s < plan->schedule.nsegments; ++s) {
        auto seg = plan->schedule.segments[s];
        // Call the specialized kernel function for this segment
        vsla_error_t err = seg.fn(plan, P->values, P->nodes, seg.first_node, seg.last_node, bind, stream); [cite: 4, 5]
        if (err != VSLA_ERR_OK) {
            return err; // Propagate error
        }
    }
    return VSLA_ERR_OK;
}
```

### 7.3. Fusion Rules
Fusion aims to reduce overhead by combining multiple IR nodes into a single, optimized kernel.

**MUST** fuse consecutive nodes if **ALL** the following conditions are true:

*   They operate on the same ambient output shape.
*   All involved operations are element-wise or simple reductions (e.g., add, subtract, Hadamard product, exponential, logarithm, mask, scalar multiplication, residual add).
*   No operation within the chain requires a different arena class or a separate workspace buffer that would break contiguity.
*   No operation acts as an implicit barrier (e.g., FFT, matrix multiplication, convolution, or other complex operations that cannot be easily inlined).

**Example fused element-wise kernel template**:
```c
static vsla_error_t seg_elemwise_chain(const vsla_plan_t* plan,
                                     const vsla_value_t* values,
                                     const vsla_node_t* nodes,
                                     size_t first_node, size_t last_node,
                                     const vsla_bindings_t* bind,
                                     vsla_stream_t* stream){
    // Resolve base pointers for inputs and outputs from the 'bind' structure and 'plan->offsets'.
    float* a = arena_ptr(bind, values[a_id].arena_class, values[a_id].arena_offset);
    float* b = arena_ptr(bind, values[b_id].arena_class, values[b_id].arena_offset);
    float* y = arena_ptr(bind, values[y_id].arena_class, values[y_id].arena_offset);

    size_t N = vsla_logical_elems(out_value_tensor); // Use the ambient size for the loop.

    #pragma omp parallel for schedule(static) // Example threading using OpenMP
    for (size_t i = 0; i < N; ++i) {
        // Operations are directly inlined based on the fused nodes.
        float tmp = a[i] + b[i];       // Corresponds to node 'first_node'
        tmp = tmp * scale;             // Corresponds to node 'first_node + 1'
        y[i] = relu(tmp);              // Corresponds to node 'last_node'
    }
    return VSLA_ERR_OK;
} [cite: 4, 5]
```

### 7.4. Streams
`vsla_stream_t` is an opaque handle representing an execution stream (e.g., a CPU thread-pool queue or a GPU device stream). All operations within a stream **SHOULD** maintain their order and execute asynchronously if supported by the backend, allowing for potential overlap of computation and data transfers.

## 8. Parallelism and Hardware Acceleration
This section outlines the strategy for parallel execution and leveraging specialized hardware.

### 8.1. CPU Multithreading
The reference CPU backend **SHOULD** be multithreaded to take advantage of modern multi-core processors.

*   **Strategy**: Loop-level parallelism is the primary mechanism for multithreading.
*   **Implementation**: OpenMP is the recommended and simplest approach for parallelizing loops within fused kernels, as shown in the example in Section 7.3.
*   **Planner Consideration**: The planner **MAY** consider the thread count when making decisions about kernel fusion or segment granularity.
*   **Streams**: For the CPU backend, a `vsla_stream_t` can represent a handle to a thread pool, allowing for more sophisticated asynchronous execution patterns.

### 8.2. GPU Acceleration (CUDA/ROCm)
A primary goal for VSLA is to provide high-performance execution on GPUs. The `vsla_backend_t` abstraction is designed specifically for this.

*   **CUDA Graphs / HIP Graphs**: The program/plan/execute model is designed to map directly to CUDA or HIP graphs. A `vsla_profile_t` for a specific shape bucket can be recorded once and replayed many times, minimizing kernel launch overhead. The `record_graph` and `replay_graph` function pointers in the vtable facilitate this.
*   **Specialized Kernels**: GPU backends **MUST** implement custom kernels for VSLA operations. These kernels should be optimized for the target architecture, including the use of shared memory and warp-level primitives.
*   **Tensor Cores / Matrix Cores**: For operations like `VSLA_OP_MATMUL` and `VSLA_OP_CONV1D`, GPU kernels **SHOULD** be written to leverage specialized hardware units (e.g., NVIDIA Tensor Cores, AMD Matrix Cores) for maximum throughput.
*   **Data Locality**: The memory planner's output (arena offsets) can be used by the GPU backend to optimize data placement and memory access patterns, potentially placing frequently used tensors in faster memory like texture caches if applicable.

## 9. Autograd (Reverse-Mode AD)
VSLA supports reverse-mode automatic differentiation through the construction and execution of a backward program (Pᵀ) that computes gradients.

### 9.1. Build Backward Program Pᵀ
The backward program is constructed by traversing the forward program's nodes in reverse topological order. For each node, the Vector-Jacobian Product (VJP) rule is applied to propagate gradients backward.

```c
// Conceptual algorithm:
// grad_map: a mapping from forward value ID to its corresponding gradient value ID in the backward program.
//           Initialized with zero-tensors (buffers) for values that will receive gradients.

for (node in reverse_topo_order(P.nodes)) { [cite: 4, 5]
    // g_outputs are the incoming gradients (cotangent vectors) for the output(s) of the forward node.
    // These are retrieved from `grad_map` based on the forward node's outputs.
    g_outputs = get_gradients_for_outputs(node.outputs, grad_map);

    // emit_vjp(node, g_outputs) generates the necessary backward nodes (ops) for this node,
    // which compute the gradients w.r.t. its inputs based on 'g_outputs'.
    // It returns a tuple of gradient values for each input of the forward node.
    vjp_gradients_for_inputs = emit_vjp_nodes(node, g_outputs); [cite: 4, 5]

    // Accumulate the computed VJP gradients into the respective input gradients in `grad_map`.
    // This handles multiple paths contributing to an input's gradient.
    for each input 'i' of 'node':
        accumulate_gradient(grad_map[i], vjp_gradients_for_inputs[i]); [cite: 4, 5]
}
```
*   `grad_map[x]` (for an input `x`) is initialized to zero-tensors (pre-allocated by the planner) or lazily materialized as needed to ensure a buffer for accumulation.
*   Parameter gradients will accumulate into allocated buffers in the `GRADS` arena.

### 9.2. Checkpointing
Checkpointing is a memory-saving technique for autograd:

*   Specific forward nodes can be explicitly marked to store their outputs (these values are allocated in the `ACTIVATIONS` arena and retained).
*   Any forward value not checkpointed will be recomputed during the backward pass if needed, by emitting the necessary forward subgraph nodes into the backward program (Pᵀ).
*   The planner for the backward pass (`Pb`) **MUST** account for these retained forward activations, ensuring their memory is correctly managed within the `ACTIVATIONS` arena.

### 9.3. Planning & Execution for Backward Pass
*   A separate `vsla_program_t` (B) is built for the backward pass (Pᵀ).
*   A `vsla_plan_t` (Pb) is generated for B. This plan **SHOULD** share the `PARAMS` arena with the forward plan (Pf) if gradients are being accumulated in-place or streamed. It **MAY** optionally share the `ACTIVATIONS` arena if retained activations are passed directly.
*   The backward program is executed after the forward pass, typically provided with the initial `loss_grads` (gradient of the loss with respect to the program's outputs).

### 9.4. Vector-Jacobian Product (VJP) Table
This table defines the backward pass for each forward operation. `g_out` denotes the gradient flowing into the op's output(s). We accumulate into `g_X` for each input `X`. `⊙` denotes Hadamard (element-wise) product.

| Operation | Forward (Y = f(X)) | Gradient w.r.t. Inputs (g_X) | Notes |
| :--- | :--- | :--- | :--- |
| **Elementwise Core** | | | |
| Add | `Y=A+B` | `gA += g_out`, `gB += g_out` | Automatic shape promotion implies padding during forward, but `unprom` (slice/extract) for gradient ensures it flows back to original dimensions. |
| Sub | `Y=A−B` | `gA += g_out`, `gB -= g_out` | |
| Hadamard | `Y=A ⊙ B` | `gA += g_out ⊙ B`, `gB += g_out ⊙ A` | |
| Mul Scalar | `Y=s ⋅ X` | `gX += s ⊙ g_out`, `gs += <g_out, X>` (reduce to scalar) | The inner product `<g_out, X>` implies a sum reduction over all elements to produce the scalar gradient `gs`. |
| Exp | `Y= exp(X)` | `gX += g_out ⊙ Y` | Uses the forward output `Y` (which is `exp(X)`) for efficiency. |
| Log | `Y= log(X)` | `gX += g_out ⊙ (1/X)` | |
| ReLU | `Y= max(0,X)` | `gX += g_out ⊙ 1_{X>0}` | `1_{X>0}` is the indicator function: 1 if `X > 0`, else 0. For `X=0`, gradient is typically 0. |
| Sigmoid | `Y= sigma(X)` | `gX += g_out ⊙ Y ⊙ (1 − Y)` | Uses the forward output `Y` for efficiency. |
| Tanh | `Y= tanh(X)` | `gX += g_out ⊙ (1 − Y²)` | Uses the forward output `Y` for efficiency. |
| GELU | `Y= GELU(X)` | `gX += g_out ⊙ GELU'(X)` | Requires implementation of the GELU derivative (`GELU'(X)`), which depends on the specific GELU approximation (e.g., `tanh` or `erf` variant). |
| Mask | `Y=X ⊙ M` (M is constant) | `gX += g_out ⊙ M` | The mask `M` is not a trainable parameter, so no gradient w.r.t. `M`. |
| Dropout | `Y=X ⊙ M/(1−p)` (M is mask, p is dropout prob) | `gX += g_out ⊙ M / (1-p)` | The random mask `M` (binary, 0 or 1) **MUST** be cached during the forward pass and reused for the backward pass. |
| **Reductions / Broadcast** | | | |
| Sum (axis=K) | `Y= sum_K X` | `gX += broadcast_to_shape(g_out, X.shape)` | The gradient `g_out` is broadcast back to the shape of `X` along the summed axes. |
| Mean (axis=K) | `Y=(1/N) sum_K X` (N is size of K axis) | `gX += broadcast_to_shape(g_out) / N` | Gradient is broadcast and scaled by `1/N`. |
| **Affine & Norms** | | | |
| LayerNorm | `Y=(X− mu)/ sigma ⋅ gamma+ beta` | Standard LayerNorm backward using saved `mu`, `sigma` or recompute. Outputs: `gX`, `gγ`, `gβ`. | |
| RMSNorm | `Y=X/ rms(X) ⋅ gamma` | `gγ += ∑ (g_out ⊙ X / rms(X))` , `gX += …` (complex formula) | The `rms(X)` (root mean square) is `sqrt(mean(X^2))`. The full `gX` derivation involves the chain rule for `1/rms(X)`. |
| **Linear Algebra** | | | |
| Matmul | `Y=A ⋅ B` | `gA += g_out ⋅ B^T`, `gB += A^T ⋅ g_out` | Standard matrix multiplication gradients. |
| Softmax (row-wise) | `Y= softmax(X)` | `gX = g_out ⊙ Y - Y ⊙ ∑_{row}(g_out ⊙ Y)` (per-row reduction) | For `Y_i = exp(X_i) / ∑_j exp(X_j)`. |
| **VSLA Semiring Ops** | | | |
| Conv1D (Model A) | `Y=A ∗conv∗ B` | `gA += g_out *corr* flip(B)`, `gB += flip(A) *corr* g_out` | Correlation is convolution with one operand flipped. FFT path **MAY** be used for efficiency. `flip(X)` reverses the vector. |
| Kronecker (Model B) | `Y= kron(A,B)` | For 1D (vectors) A (length m), B (length n): Reshape `g_out` to (m, n) blocks. Then: `gA[i] += ∑_j g_out[i,j] ⋅ B[j]`<br>`gB[j] += ∑_i A[i] ⋅ g_out[i,j]` | This is equivalent to summing the outer product of `g_out` with `B` for `gA`, and `A` with `g_out` for `gB`. More generally, for tensors, it involves distributing the gradient to the correct blocks and summing. |
| **Structural** | | | |
| Stack | `Y= Stack(X_1, ...,X_k)` | `gX_i += Slice(g_out, i)` | The `Slice` operation extracts the i-th tensor along the stacking dimension from `g_out` and unpromotes it to `X_i`'s original shape. |
| Window | `Y= Stack(last w inputs)` | Propagate slices back to each contributing input from the window. | This requires tracking which input tensors contributed to each window. |
| Reshape/Transpose/Slice/Gather | Shape-only transforms | `gX = inverse_transform(g_out)` | These operations primarily affect shape metadata. The backward pass involves applying the inverse shape transformation to the incoming gradient. |
| Gather | `Y=X[ indices]` | `gX += scatter(g_out, indices)` (else 0) | `gX` is a sparse tensor where gradients `g_out` are scattered back to their original positions defined by `indices` ; other positions are zero. |

**Implementation Notes for VJP Table**:
*   You **MUST** implement reduction operations for scalar gradients (e.g., `gs` for `MulScalar`).
*   For broadcast/ambient promotion gradients, you **MUST** sum-reduce over the broadcasted axes to correctly accumulate gradients back to the original input shape.

## 10. Cross-Program / Cross-Arena Composition
VSLA programs **MUST** be able to compose and share data effectively, especially for complex models or distributed systems.

### 10.1. Import/Export Values (`vsla_link_t`)
```c
typedef struct {
    const vsla_program_t* src;        // Source program from which a value is exported
    uint32_t              src_value_id; // ID of the value in the source program
    vsla_program_t*       dst;        // Destination program to which the value is imported
    uint32_t              dst_value_id; // ID of the value in the destination program
} vsla_link_t; [cite: 5]
```
*   When a `vsla_link_t` is established, the `dst_value_id` in the destination program gets `arena_class = VSLA_ARENA_EXTERNAL`.
*   The actual memory pointer and offset for `VSLA_ARENA_EXTERNAL` values **MUST** be supplied at bind time (`vsla_program_bind`).
*   The caller is responsible for enforcing correct sequencing of program executions (e.g., source program completes before destination program starts) or using backend-specific stream/event synchronization if supported.

### 10.2. Shared Arena Manager
A higher-level Shared Arena Manager **MAY** be implemented. This manager would pre-allocate a large, unified memory region (a "mega-arena") for all programs.

In such a setup, individual program plans would be colored in a global pass by the manager, ensuring all offsets are relative to the mega-arena. This allows seamless data sharing and reduces fragmentation across multiple programs.

## 11. Backend Abstraction (C vtable)
VSLA's architecture supports multiple hardware backends (CPU, GPU, etc.) via a C vtable, enabling device-agnostic program definition.

```c
typedef struct vsla_backend_vtable_s {
    // Memory management functions
    void* (*alloc)(size_t bytes, size_t alignment, void* ctx);           // Allocates device memory
    void  (*free)(void* ptr, void* ctx);                                 // Frees device memory
    void  (*memcpy_h2d)(void* dst, const void* src, size_t bytes, void* ctx); // Host to device memory copy
    void  (*memcpy_d2h)(void* dst, const void* src, size_t bytes, void* ctx); // Device to host memory copy
    void  (*memset_zero)(void* dst, size_t bytes, void* ctx);            // Fills device memory with zeros

    // Execution functions
    vsla_error_t (*launch_segment)(const vsla_segment_t* segment,
                                 const vsla_plan_t* plan,
                                 vsla_stream_t* stream,
                                 void* ctx);                             // Launches a fused segment on device
    vsla_error_t (*synchronize)(vsla_stream_t* stream, void* ctx);        // Synchronizes the stream

    // Optional graph recording/replay functions (e.g., CUDA Graphs)
    vsla_error_t (*record_graph)(void* program_handle, vsla_stream_t* stream, void* ctx, void** graph_handle_out); // Records execution graph
    vsla_error_t (*replay_graph)(void* graph_handle, vsla_stream_t* stream, void* ctx); // Replays recorded graph
} vsla_backend_vtable_t; [cite: 5]

typedef struct {
    const vsla_backend_vtable_t* vtbl; // Pointer to the backend's function table
    void*                        ctx;  // Backend-specific context handle (e.g., CUDA device ID)
} vsla_backend_t; [cite: 5]
```
A CPU backend **MUST** provide a correct and simple baseline implementation for all required functions.

## 12. Public C API (End-to-End Lifecycle)
This section illustrates the typical lifecycle of a VSLA computation using the public C API.

### 12.1. Compile (Forward Pass)
```c
// 1. Build the program (IR) from a user-defined graph description.
vsla_program_t* P = vsla_program_build(&graph_description, &build_options);
if (!P) { /* Handle error from vsla_get_last_error() */ }

// 2. Plan the program for a specific backend (memory layout, execution schedule).
vsla_plan_t* Pf = vsla_program_plan(P, &plan_options, backend);
if (!Pf) { /* Handle error */ vsla_program_destroy(P); }

// 3. Bind the plan to allocate arenas and set up value offsets.
vsla_bindings_t bf = {0}; // Structure to hold arena pointers
vsla_error_t err = vsla_program_bind(P, Pf, backend, &bf);
if (err != VSLA_ERR_OK) { /* Handle error */ vsla_plan_destroy(Pf); vsla_program_destroy(P); }
// At this point, all necessary memory for the forward pass is allocated and mapped.
```

### 12.2. Execute (Forward Pass)
```c
// Prepare input data (e.g., copy from host to device if not CPU backend).
vsla_io_t inputs = { /* ... populate with input tensor data */ };
vsla_io_t outputs = { /* ... populate with output tensor placeholders */ };
vsla_stream_t* stream = vsla_backend_create_stream(backend); // Create an execution stream

// Execute the forward pass. This call MUST NOT allocate memory.
vsla_error_t err = vsla_program_execute(P, Pf, &bf, &inputs, &outputs, stream);
if (err != VSLA_ERR_OK) { /* Handle error */ }

vsla_backend_synchronize(stream, backend->ctx); // Ensure all operations complete
// Results are now in 'outputs' buffers.

// Release the stream (if applicable)
vsla_backend_destroy_stream(stream);
```

### 12.3. Compile & Run (Backward Pass)
```c
// 1. Build the backward program (Pᵀ) from the forward program.
// This involves generating VJP nodes based on the forward graph.
vsla_program_t* B = vsla_autograd_build_backward(P);
if (!B) { /* Handle error */ }

// 2. Plan the backward program. This plan MAY share PARAMS and ACTIVATIONS arenas with Pf.
vsla_plan_t* Pb = vsla_autograd_plan_backward(B, Pf, &ad_options, backend);
if (!Pb) { /* Handle error */ vsla_program_destroy(B); }

// 3. Bind the backward plan to allocate its arenas (e.g., GRADS arena)
// and set up value offsets.
vsla_bindings_t bb = {0};
vsla_error_t err_b = vsla_program_bind(B, Pb, backend, &bb);
if (err_b != VSLA_ERR_OK) { /* Handle error */ }
// Note: bb.params_arena_ptr MAY be set to bf.params_arena_ptr for shared parameter storage.

// Prepare initial loss gradients and storage for parameter gradients.
vsla_io_t loss_grads = { /* ... populate with loss gradient data */ };
vsla_io_t param_grads = { /* ... populate with parameter gradient placeholders */ };
vsla_stream_t* b_stream = vsla_backend_create_stream(backend);

// Execute the backward pass. This call MUST NOT allocate memory.
err_b = vsla_autograd_execute(B, Pb, &bb, &inputs, &outputs, &loss_grads, &param_grads, b_stream);
if (err_b != VSLA_ERR_OK) { /* Handle error */ }

vsla_backend_synchronize(b_stream, backend->ctx); // Synchronize backward stream
// Parameter gradients are now available in 'param_grads'.

// Clean up
vsla_backend_destroy_stream(b_stream);
vsla_program_destroy(B);
vsla_plan_destroy(Pb);
```

### 12.4. Profiles Usage
```c
// 1. Compile a profile for a specific shape bucket (e.g., seq_len = 2048).
// This internally builds the program, plans it, and binds arenas.
vsla_profile_t* prof = vsla_profile_compile(&graph_description, &bucket_2048_config, backend);
if (!prof) { /* Handle error */ }

// 2. Execute using the profile. This call is highly optimized and MUST NOT re-allocate.
// The profile manages its internal program, plan, and bindings.
vsla_error_t err_prof = vsla_profile_run(prof, &inputs, &outputs, stream);
if (err_prof != VSLA_ERR_OK) { /* Handle error */ }

// The profile can be run repeatedly for any inputs matching 'bucket_2048_config'.
// Clean up the profile when no longer needed.
vsla_profile_destroy(prof);
```

## 13. Error Handling and Reporting
A robust error handling strategy is essential for a production-ready library.

### 13.1. Error Codes (`vsla_error_t`)
All public API functions that can fail **MUST** return a `vsla_error_t` enum value.

```c
typedef enum {
    VSLA_ERR_OK = 0,
    VSLA_ERR_OOM,                // Out of memory
    VSLA_ERR_INVALID_ARG,        // An invalid argument was provided
    VSLA_ERR_INVALID_SHAPE,      // Incompatible or invalid tensor shapes for an operation
    VSLA_ERR_INVALID_PROGRAM,    // Program validation failed
    VSLA_ERR_PLANNING_FAILED,    // Memory planner could not produce a valid plan
    VSLA_ERR_BACKEND_ERROR,      // An error occurred within the backend (e.g., CUDA error)
    VSLA_ERR_NOT_IMPLEMENTED,    // The requested operation is not implemented for the backend
    VSLA_ERR_UNKNOWN,            // An unknown error occurred
} vsla_error_t;
```

### 13.2. Retrieving Error Information
To provide more context than a simple error code, the API **SHOULD** provide a way to retrieve a detailed, human-readable error message. This is best handled via a thread-local storage mechanism.

```c
// Returns a string describing the last error that occurred on the calling thread.
// The returned pointer is valid until the next API call on the same thread.
const char* vsla_get_last_error_string();
```

### 13.3. Best Practices
*   **Always check return values**: Callers **MUST** check the `vsla_error_t` return code of every API function.
*   **Retrieve detailed message**: If an error occurs, the caller **SHOULD** immediately call `vsla_get_last_error_string()` to get more information.
*   **No `errno`**: The VSLA library **MUST NOT** use the global `errno` variable.

## 14. Canonical Applications and Examples

### 14.1. Transformer Worked Example
This example illustrates memory requirements for a Transformer model. Assume a model with `L` layers, `d_model` embedding dimension, `h` attention heads, `d_head` head dimension, `d_ff` feed-forward dimension, `V` vocabulary size, `seq_len` bucket `S`, and `batch_size` `B`.

#### 14.1.1. Parameter Memory (`VSLA_ARENA_PARAMS`)
Total parameters (`P`) = sum of:
*   **Embeddings**: `V * d_model`
*   **QKV projections** (per layer): `d_model * 3 * d_model` (for Q, K, V linear layers)
*   **Attention output** (per layer): `d_model * d_model`
*   **Feed-Forward** (2 linear layers per layer): `d_model * d_ff + d_ff * d_model`
*   **LayerNorm/RMSNorm** (per layer): `2 * d_model` (for `γ, β` if LayerNorm, or `d_model` if RMSNorm for `γ` only)

Total `PARAMS` bytes = `P * sizeof(dtype)`. This arena is shared across all sequence length buckets.

#### 14.1.2. Activation Memory (`VSLA_ARENA_ACTIVATIONS`)
Rough upper bound for the forward pass (without checkpointing):
*   **Token embeddings**: `B * S * d_model`
*   **Q, K, V tensors**: `3 * B * S * d_model`
*   **Attention scores**: `B * h * S * S` (for dense full attention)
*   **Attention probabilities**: `B * h * S * S`
*   **Attention output**: `B * S * d_model`
*   **MLP intermediates**: `B * S * d_ff`
*   **Residuals/norms**: roughly `O(B * S * d_model)` per layer (these can often be overlapped/reused by the planner).

The planner will significantly reduce the total `ACTIVATIONS` memory via liveness coloring, by re-using memory for tensors that are no longer needed.

#### 14.1.3. Workspace Memory (`VSLA_ARENA_WORKSPACE`)
This arena holds transient buffers.
*   FFT buffers (if convolution operations are used).
*   Softmax temporary reductions.

`WORKSPACE` bytes = Maximum over all nodes of (workspace needed while that node's live window intersects others). The linear-scan memory planner will compute this peak usage.

#### 14.1.4. Gradients Memory (`VSLA_ARENA_GRADS`)
*   **`GRADS` params**: Typically the same size as `PARAMS` arena (unless gradients are streamed and updated in-place without full materialization).
*   **`GRADS` activations**: Similar magnitude to forward activations, unless aggressive checkpointing is used to recompute rather than store them.

#### 14.1.5. Profiles
*   Generate (`Program`, `Plan`) for each sequence length `S` in a defined set (e.g., `{512, 1024, 2048, 4096}`).
*   Store `arena_size[class]` for each bucket. At runtime, inputs are routed to the appropriate `S` bucket, and the pre-computed plan is bound and executed.

**Example Memory Report (for S=2048)**:
*   **`PARAMS`**: X GiB (shared across all buckets)
*   **`ACTIVATIONS` (fwd)**: Y GiB
*   **`WORKSPACE`**: Z MiB
*   **`GRADS`**: G GiB

### 14.2. Tensor Pyramid Construction
The paper's concept of a Tensor Pyramid for multi-resolution analysis can be implemented by composing `VSLA_OP_WINDOW` and `VSLA_OP_STACK`.

*   **`VSLA_OP_WINDOW`**: This operation is the first step. It takes a stream of input tensors (e.g., a long time-series vector or a set of images) and groups them into overlapping or non-overlapping windows. Each window is then stacked into a single tensor of rank `r+1`. For example, applying a `VSLA_OP_WINDOW` with `window_size=4` to a stream of 100 images would produce a stream of 25 tensors, where each tensor has the shape `[4, H, W, C]`.
*   **Recursive Stacking**: The output stream from a `VSLA_OP_WINDOW` operation can be fed into another `VSLA_OP_WINDOW` or `VSLA_OP_STACK` operation. This recursive composition builds the pyramid.
    *   **Level 0**: The initial stream of data (e.g., `[T_1, T_2, ..., T_N]`).
    *   **Level 1**: `W_1 = VSLA_OP_WINDOW(Level 0, window_size=w1)`. This produces a new, shorter stream of higher-rank tensors.
    *   **Level 2**: `W_2 = VSLA_OP_WINDOW(Level 1, window_size=w2)`. This further reduces the stream length and increases the rank.
*   This process continues until the desired pyramid height is reached. The result is a hierarchical representation of the original data, which is useful for feature extraction at multiple scales.

## 15. Numerics & Precision Policy
*   **Accumulation**: All reductions and FFT temporaries **MUST** use `double` precision for intermediate accumulations to maintain numerical stability.
*   **Storage**: Tensor elements **MAY** be stored in `float32`, `float16`, or `bfloat16`. Downcasting from `double` occurs upon storing results.
*   **FMA**: Use `fma` (fused multiply-add) instructions where available (e.g., convolution inner loops) for improved precision and performance.
*   **Comparison Tolerance**: For floating-point comparisons in tests, use `abs(x-y) <= atol + rtol*max(|x|,|y|)`. Recommended default tolerances: `(atol, rtol) = (1e-12, 1e-9)` for `double` and relaxed values for `float` (e.g., `1e-6, 1e-3` for `fp32`).
*   **FFT Error**: Forward/inverse FFT operations have an error bound of `O(ε log L)` (where `ε` is machine epsilon). The inverse FFT **MUST** scale its output by `1/L`. Double temporaries help reduce cancellation errors.
*   **Flush-to-zero**: An optional global toggle **MAY** be provided to flush denormal numbers to zero for performance, but this can impact numerical exactness.
*   **Empty Tensors**: All kernels **MUST** guard against `data==NULL` dereferences and early-exit gracefully when encountering empty tensors (logical size = 0).

## 16. Testing Matrix
Comprehensive testing is crucial for ensuring correctness and performance across all VSLA components.

| Category | MUST Pass Checks |
| :--- | :--- |
| **IR Build & Validation** | Topological sort succeeds; invalid dtypes are rejected; shapes are correctly inferred for all ops; `capacity >= shape` is enforced. |
| **Planning** | No overlapping memory intervals within an arena; all offsets are 64-byte aligned; arena sizes are stable across identical planning runs. |
| **Determinism** | N identical runs produce bit-for-bit identical results; zero dynamic allocations are observed during `execute()` (verify with `malloc`/`free` overrides). |
| **Autograd Core** | Finite-difference gradient checks on random graphs and shapes: maximum relative error **MUST** be below the defined tolerance. |
| **Profiles** | Selecting different buckets **MUST NOT** trigger re-planning or re-allocation. |
| **Backends** | CPU vs. GPU numerical parity **MUST** be within the defined tolerance. |
| **Cross-Program** | `vsla_link_t` import/export mechanisms work correctly; external arena offsets are valid. |
| **Fuzz Testing** | Randomly generated DAGs (within bounded size), shapes, and dtypes are processed correctly for `10^4` or more cases. |
| **Unit Tests** | * **Addition/Subtraction**: `[1,2,3] + [4,5] -> [5,7,3]`; subtraction producing trailing zeros correctly shrinks.<br> * **Hadamard**: Correct element-wise product.<br> * **Kronecker**: Non-commutativity `kron([1,2],[0,1])` vs. reversed is verified.<br> * **Convolution**: Direct vs. FFT path equivalence within tolerance for sizes above threshold; FFT plan cache reuse; empty operands; shrink correctness.<br> * **Stacking**: Heterogeneous shapes stack correctly; ambient promotion is verified.<br> * **Window Pyramid**: Feed `2^L` tensors; verify cascade height and outputs.<br> * **Empty Operands**: All operations handle empty operands gracefully and produce correct empty/zero results.<br> * **FFT Specific**: Plan cache reuse; numerical accuracy; edge cases for `L` (e.g., 1, 2, 4). |
| **Property-Based Testing** | In addition to the existing property tests, the following should be added:<br> * **Idempotence**: `shrink(shrink(T)) == shrink(T)`.<br> * **Equivalence**: `(T + Z) == T` where `Z` is a zero tensor.<br> * **Shape Invariance**: `shape(T + Z) == shape(T)`.<br> * **Rank Invariance**: `rank(T + Z) == rank(T)`.<br> * **Preservation of Trailing Zeros**: `(T + Z)` has the same number of trailing zeros as `T`. |

## 17. Performance Checklist
These are critical guidelines for achieving high performance in a VSLA implementation.

*   **Fuse Elementwise Chains**: Implement segment fusion for element-wise operations to reduce kernel launch overhead.
*   **64-byte Alignment**: All memory allocations and tensor offsets **MUST** adhere to 64-byte alignment to enable optimal SIMD and cache performance.
*   **Cache FFT Plans**: FFT plans (twiddles, bit-reversal tables) **MUST** be cached by FFT length (`L`) to avoid redundant computation.
*   **Persistent Kernels / CUDA Graphs**: For GPU backends, leverage persistent kernels or CUDA Graphs to minimize driver overhead for repeated executions.
*   **NUMA-aware Partitions**: (Optional) For large-scale CPU deployments, consider NUMA-aware memory placement strategies.
*   **Profile-Guided Specialization**: The planner **MAY** use profiling information to specialize the plan (e.g., grouping hot tensors for better cache locality).
*   **No Dynamic Allocations in `execute()`**: This is a fundamental invariant. Ensure that runtime execution paths perform no `malloc` or `free` calls.

## 18. Migration Guide (v3.2 → v4.3)
This guide outlines the transition process from the v3.2 API to the v4.3 program/plan/execute architecture.

*   **Keep v3.2 Primitive Kernels**: The underlying arithmetic (`vsla_add`, `vsla_conv`, `vsla_kron`, etc.) from v3.2 can be largely retained. These will now be called from within the new `vsla_segment_fn` implementations defined by the executor.
*   **Wrap Legacy APIs**: The old per-operation APIs (`vsla_add(out, a, b)`) **SHOULD** be wrapped as single-operation `vsla_program_t` objects for backward compatibility. This allows existing v3.2 code to gradually migrate.
*   **Incremental Autograd**: Implement the autograd VJP table incrementally, starting with core operations, using the detailed specifications in Section 9.4.
*   **Adopt New Codepaths**: New development **SHOULD** exclusively use the program, plan, and execute interfaces to leverage the architectural benefits.
*   **Memory Model Transition**: Understand that `vsla_tensor_t`s no longer manage their own memory independently via `refcnt` (though `refcnt` might still be used for debug or external ownership). Memory ownership shifts to `vsla_plan_t` and `vsla_bindings_t`. Manual `vsla_retain`/`vsla_release` on tensors will largely be replaced by the planner's allocation/deallocation of arenas.

## 19. Future Work
*   **Dynamic Shapes**: Support dynamic shapes within a single `vsla_program_t` through guarded sub-allocators or dynamic re-planning on shape changes (more sophisticated than current bucketing).
*   **Operator Autotuning**: Implement operator autotuning mechanisms (e.g., for FFT thresholds, tiling strategies, block sizes) to adapt to specific hardware and input characteristics.
*   **Compiler Passes**: Develop IR-level compiler passes for optimizations such as Common Subexpression Elimination (CSE), Dead Code Elimination (DCE), and layout optimization.
*   **Distributed Systems**: Extend arena and execution models to support distributed/multi-device computing environments, building on the parallelism strategies outlined in Section 8.
*   **Higher-Level Language Bindings**: Build robust higher-level language bindings (e.g., Python, Rust) atop the stable C ABI.
*   **Automated Broadcast-Reduction Gradient Generation**: Develop a mechanism to automatically generate broadcast/reduction gradients in the VJP rules to reduce boilerplate and potential errors.
*   **Higher-Rank Convolution**: Progress the implementation and detailed specification for axis-wise higher-rank convolution operations.
*   **Sparse Extension (Interior Zeros)**: Investigate and formalize mechanisms for handling interior (non-trailing) zeros efficiently within the VSLA framework, potentially via masks or specialized sparse formats.
*   **Categorical Formulation**: A complete categorical treatment of VSLA as a semiring-enriched category could provide deeper structural insights and new optimization opportunities.
*   **Topological Considerations**: Formalizing the preservation of topological properties (e.g., connectivity, causality) could enable certified correctness for safety-critical applications.
*   **Information-Theoretic Analysis**: Investigate fundamental limits on compression through variable-shape representations and the relationship between shape entropy and computational complexity.
*   **Synergies with Semi-Tensor Product (STP)**: Explore hybrid algebraic models where VSLA's equivalence classes integrate with STP's logical and control-theoretic tools.
*   **Algebraic Invariants for Variable-Shape Tensors**: Inspired by recent advances in the Semi-Tensor Product (e.g., DK-STP), investigate the possibility of defining algebraic invariants such as determinants or eigenvalues for variable-shape tensors within the VSLA framework. This could provide powerful new tools for analyzing and optimizing complex systems.
*   **Quantum Computing Applications**: Investigate VSLA's suitability for quantum simulation, hybrid quantum-classical optimization, and efficient simulation of quantum error correction codes due to its variable-dimensional nature.
*   **Edge Computing and Distributed Systems**: Develop ultra-low-power implementations and federated learning protocols leveraging VSLA's memory efficiency for resource-constrained environments.
*   **Domain-Specific Applications**: Continue exploring applications in computational biology, climate modeling, and financial engineering, adapting VSLA to their unique data structures and computational needs. The success of the related STP framework suggests that **Game Theory** and **Boolean Network Analysis** are also promising application areas.

End of v4.3 Document. This guide provides the authoritative specification for the VSLA C implementation. Adherence to its norms, especially the "**MUST**" and "**SHOULD**" clauses, is critical for achieving correct, high-performance, and maintainable code.

## 1. Core Concepts (precise glossary)
*   **IR (Intermediate Representation)**: SSA-like directed acyclic graph (DAG) of Nodes (operations) producing Values (tensors). Topologically sortable.
*   **Program**: A frozen, validated IR along with metadata such as constants, parameters, and input/output signatures.
*   **Plan**: The deterministic execution schedule and memory mapping for a Program, which includes: value liveness information, colored buffers, arena sizes, FFT plan cache, and a list of fused segments.
*   **Arena**: A single contiguous block of memory designated for one class of tensors (e.g., `PARAMS`, `ACTIVATIONS`, `WORKSPACE`, `GRADS`). Values point to fixed offsets within their assigned arena.
*   **Executor**: The component that runs a Program according to its Plan on a specific Backend.
*   **Backend**: A swappable device implementation (e.g., CPU, CUDA, HIP/RoCm, Metal, Vulkan, OpenCL) exposed via a C vtable.
*   **Profile (Bucket)**: A pre-compiled set consisting of a Program, its Plan, and an Executor, optimized for a specific shape family (e.g., `seq_len = 1024`). This allows for repeated runs without reallocation.
*   **Tape**: An autograd side-structure that captures the subset of forward values necessary for computing the backward pass.
*   **Checkpoint**: A forward node explicitly marked to store its outputs; other intermediate values may be recomputed during the backward pass if not checkpointed.
*   **Minimal Representative**: Each equivalence class of variable-shape vectors has a unique representative whose shape has no trailing zero hyperplanes. Runtime storage must always use minimal representatives unless explicitly stated (e.g., before `vsla_shrink` or serialization).
*   **Ambient Promotion**: The process where operands are implicitly coerced to a common dimension (the element-wise maximum of their logical dimensions) before computation, while preserving sparsity and algebraic properties. This avoids materializing padded zeros.
*   **Capacity**: The physical memory allocated for a tensor, which is typically `next_pow2(shape[i])` for each dimension `i`, and always `>= shape[i]`.
*   **Modal verbs (RFC 2119 style)**:
    *   **MUST** — mandatory for correctness.
    *   **SHOULD** — strong recommendation for performance or safety.
    *   **MAY** — optional/implementation detail.

## 2. High-Level Architecture
The VSLA system follows a distinct compilation and execution pipeline to ensure high performance and efficient resource management.

User Graph → Build IR → Infer Shapes/Capacity → Validate → Plan Memory (Liveness + Coloring) → Produce Arenas & Offsets → Lower to Executor (CPU ref) / Record CUDA Graph → Execute
↘
Build Backward IR (Autograd) → Plan → Execute

**Key Invariants**:
*   **MUST NOT** allocate memory during the `execute()` hot path.
*   Plans **MUST** be reusable without modification for every batch within the same profile.
*   Programs **MUST** be immutable after planning; rebuild the program to change shapes or operations.

## 3. IR (Intermediate Representation)
The IR provides a formal, machine-readable representation of the computation graph.

### 3.1. Data structures
```c
typedef enum {
    VSLA_TENSOR_PARAM,   // Parameter (weight/bias)
    VSLA_TENSOR_INPUT,   // Program input
    VSLA_TENSOR_OUTPUT,  // Program output
    VSLA_TENSOR_TEMP,    // Internal intermediate value
} vsla_valuetag_t; [cite: 4, 5]

typedef enum {
    VSLA_ARENA_PARAMS = 0,    // Parameters, typically read-only during forward pass
    VSLA_ARENA_ACTIVATIONS = 1, // Intermediate activations (retained for backward if not checkpointed)
    VSLA_ARENA_WORKSPACE = 2, // Scratch memory for transient computations (e.g., FFT buffers)
    VSLA_ARENA_GRADS = 3,     // Gradients w.r.t. parameters and activations
    VSLA_ARENA_EXTERNAL = 4,  // Memory managed externally, linked via bindings
} vsla_arenaclass_t; [cite: 4, 5]

typedef struct {
    uint32_t        id;            // SSA id: Unique identifier for the value within the program
    vsla_dtype_t    dtype;         // Data type of the tensor elements (e.g., float32, float64)
    uint8_t         rank;          // Number of axes (dimensions) of the tensor
    uint64_t*       shape;         // Minimal logical dimensions of the tensor (length == rank)
    uint64_t*       capacity;      // Physical allocation dimensions, typically next_pow2(shape[i]) (length == rank)
    vsla_valuetag_t tag;           // Classification of the value (parameter, input, output, temp)
    vsla_arenaclass_t arena_class; // The memory arena this value belongs to
    uint64_t        arena_offset;  // Byte offset within its assigned arena (filled by planner)
} vsla_value_t; [cite: 4, 5]

// Ops (extendable) — all are *pure* functions of their inputs
// (side effects exist only for PARAM writes during bind/load).
typedef enum {
    VSLA_OP_ADD, VSLA_OP_SUB, VSLA_OP_HADAMARD, [cite: 4, 5]
    VSLA_OP_CONV1D, VSLA_OP_KRONECKER, [cite: 4, 5]
    VSLA_OP_SOFTMAX, VSLA_OP_LAYERNORM, VSLA_OP_RMSNORM, [cite: 4]
    VSLA_OP_MATMUL, [cite: 4]
    VSLA_OP_RELU, VSLA_OP_GELU, VSLA_OP_SIGMOID, VSLA_OP_TANH, [cite: 4]
    VSLA_OP_DROPOUT, [cite: 4]
    VSLA_OP_SUM, VSLA_OP_MEAN, [cite: 4]
    VSLA_OP_STACK, VSLA_OP_WINDOW, [cite: 4]
    VSLA_OP_RESHAPE, VSLA_OP_TRANSPOSE, VSLA_OP_SLICE, VSLA_OP_GATHER, VSLA_OP_SCATTER, [cite: 4]
    VSLA_OP_MASK, VSLA_OP_EXP, VSLA_OP_LOG, VSLA_OP_MUL_SCALAR, [cite: 4]
    VSLA_OP_PARAMETER, VSLA_OP_INPUT, VSLA_OP_OUTPUT, [cite: 4, 5]
} vsla_opkind_t; [cite: 4, 5]

typedef struct vsla_node_s {
    uint32_t        id;          // Unique identifier for the node within the program
    vsla_opkind_t   kind;        // Type of operation this node performs
    vsla_model_t    model;       // VSLA_MODEL_A or VSLA_MODEL_B for semiring ops; ignore if N/A
    uint32_t*       inputs;      // Array of value IDs that are inputs to this node
    uint32_t        ninputs;     // Number of inputs
    uint32_t*       outputs;     // Array of value IDs that are outputs from this node
    uint32_t        noutputs;    // Number of outputs
    void*           attrs;       // Pointer to op-specific attributes (e.g., axis for reduction, epsilon for BatchNorm)
} vsla_node_t; [cite: 4, 5]

typedef struct {
    vsla_node_t*  nodes;            // Array of nodes in topological order
    size_t        nnodes;           // Number of nodes
    vsla_value_t* values;           // Array of all values (tensors) in the program
    size_t        nvalues;          // Number of values
    uint32_t*     program_inputs;   // Array of value IDs that are external inputs to the program
    size_t        n_program_inputs; // Number of program inputs
    uint32_t*     program_outputs;  // Array of value IDs that are external outputs of the program
    size_t        n_program_outputs; // Number of program outputs
} vsla_program_t; [cite: 4, 5]
```

### 3.2. Validation Checklist (MUST)
Before planning or execution, a `vsla_program_t` **MUST** pass these validation checks:

*   The graph **MUST** be acyclic (a topological order **MUST** exist).
*   All node inputs and outputs **MUST** refer to existing `vsla_value_t` IDs.
*   Data types (`dtype`) **MUST** match per the operation's rules (e.g., all inputs to an `ADD` operation **MUST** have compatible dtypes).
*   The `vsla_model_t` (A or B) **MUST** be consistent where required for semiring operations.
*   No rank-0 tensors **MUST** be visible at the API boundary (internal handling may allow for rank==0 but public API should avoid it).
*   All shapes **MUST** be computed and non-negative.
*   `capacity[i]` **MUST** be `>= shape[i]` for all dimensions `i`.
*   All internal shape multiplications **MUST** pass overflow guards to prevent `UINT64_MAX` overflow.

## 4. Shape / Capacity Inference
Shape inference determines the logical dimensions of output tensors based on input shapes. Capacity inference determines the physical allocation size for each dimension.

### 4.1. Rules
Each operation defines its output shape purely as a function of its input shapes.

The logical region of a tensor, from index 0 up to `shape[i]-1` for each dimension, **MUST** be fully initialized.

The slack region (indices `>= shape[i]` but `< capacity[i]`) **MAY** contain uninitialized data and **MUST NOT** be read.

```c
for (node in topo_sorted(program.nodes)) {
    // Assume 'out' is the primary output value of the node, and 'a', 'b' are its primary inputs.
    // Full rules for all ops will be implemented.
    switch(node.kind) {
        case VSLA_OP_ADD: case VSLA_OP_SUB: case VSLA_OP_HADAMARD:
            // Element-wise operations: output shape is the element-wise maximum of input shapes (ambient shape).
            out.shape[i] = max(a.shape[i], b.shape[i]);
            break;
        case VSLA_OP_CONV1D:
            // Convolution (Model A): For vectors of length m and n, output length is m+n-1.
            out.shape[0] = (m == 0 || n == 0) ? 0 : (m + n - 1); [cite: 3]
            break;
        case VSLA_OP_KRONECKER:
            // Kronecker Product (Model B): For vectors of length m and n, output length is m*n.
            out.shape[0] = (m == 0 || n == 0) ? 0 : (m * n); [cite: 3]
            break;
        case VSLA_OP_STACK:
            // Stacking operator: rank increases by 1. New leading dimension is k (number of stacked tensors).
            // Remaining dimensions are the ambient shape of inputs.
            out.rank = input[0].rank + 1;
            out.shape[0] = k; // Number of stacked tensors
            for (i = 0; i < input[0].rank; ++i) {
                out.shape[i+1] = max_i(input[j].shape[i]); // element-wise max over all inputs for other dimensions
            }
            break;
        // ... (rules for other operations like Softmax, Matmul, Reshape, etc.)
    }
    // Capacity policy: For each dimension, capacity is the smallest power of two greater than or equal to the shape.
    // The planner MAY override this to shape[i] for small or static dimensions if it leads to better packing.
    for i in axes: value.capacity[i] = next_pow2(value.shape[i]); [cite: 5]
}
```

### 4.2. FFT Workspace Sizing
For convolution operations (`VSLA_OP_CONV1D`), FFTs require specific buffer lengths.

*   For vector-vector convolution (Model A), the FFT length `L` is `next_pow2(m + n - 1)`.
*   For higher-rank tensor convolution (iterating over element pairs), a single `L` is computed as `next_pow2(2 * d_max - 1)` where `d_max` is the maximum vdim across all elements (`T_ij`) involved in the matrix operation, ensuring enough room for the full convolution output.
*   FFT plans for a given `L` **SHOULD** be memoized (cached) to avoid redundant plan creation.

## 5. Memory Planning
Memory planning is a crucial phase that ensures efficient memory utilization and the "no allocation in hot path" invariant. It involves determining the lifetime of each value and assigning it an offset within a pre-allocated arena.

### 5.1. Data Structures
```c
typedef struct {
    uint32_t value_id;     // ID of the vsla_value_t
    uint32_t start;        // First node index (inclusive) where the value is live
    uint32_t end;          // Last node index (inclusive) where the value is live
    size_t   bytes;        // Size of the value in bytes (product(capacity) * sizeof(dtype))
    vsla_arenaclass_t arena_class; // Which arena class this value belongs to
} vsla_interval_t; [cite: 4, 5]

typedef struct {
    uint64_t offset;       // Byte offset in arena
    uint64_t size;         // Length of this memory block
} vsla_block_t; [cite: 4]

// Represents the complete memory layout and execution schedule for a program.
typedef struct {
    uint64_t          arena_size[5]; // Total size in bytes for each arena class (PARAMS, ACTIVATIONS, WORKSPACE, GRADS, EXTERNAL)
    uint64_t*         offsets;       // Array parallel to program->values, storing byte offset within its arena
    vsla_schedule_t   schedule;      // Fused segments and their execution order (see Section 7.1)
    // Additional caches and metadata, e.g., FFT plan cache.
} vsla_plan_t; [cite: 4, 5]
```

### 5.2. Liveness Computation
Liveness analysis determines the lifetime (first and last use) of each value in the topologically sorted program.

```c
for (v in program->values) {
    v.first_use = UINT32_MAX; // Initialize to maximum possible value
    v.last_use  = 0;          // Initialize to minimum possible value
}

for (idx = 0; idx < program->nnodes; ++idx) {
    vsla_node_t* node = &program->nodes[idx];
    for (i = 0; i < node->noutputs; ++i) {
        uint32_t out_id = node->outputs[i];
        program->values[out_id].first_use = min(program->values[out_id].first_use, idx);
        program->values[out_id].last_use  = max(program->values[out_id].last_use, idx);
    }
    for (i = 0; i < node->ninputs; ++i) {
        uint32_t in_id = node->inputs[i];
        program->values[in_id].last_use   = max(program->values[in_id].last_use, idx);
    }
} [cite: 4, 5]
```
**Note**: `first_use` for program inputs (e.g., `VSLA_TENSOR_INPUT`, `VSLA_TENSOR_PARAM`) should be 0.

### 5.3. Linear-Scan Buffer Coloring
This algorithm assigns contiguous memory blocks (offsets) to values within their respective arena classes, minimizing total memory footprint by reusing space from dead values.

```c
for (arena_class in {VSLA_ARENA_PARAMS, VSLA_ARENA_ACTIVATIONS, VSLA_ARENA_WORKSPACE, VSLA_ARENA_GRADS}) { [cite: 4, 5]
    intervals = collect_intervals(program->values, arena_class); // Build interval_t for relevant values
    sort(intervals by start_node_idx asc, end_node_idx asc); // Sort intervals by start time, then end time

    free_list = min_heap_of_blocks(); // Store available free blocks, sorted by offset
    arena_tail = 0; // Current end of the arena, growing as new blocks are allocated

    for (it in intervals) {
        // 1) Expire dead blocks: Remove blocks from free_list whose end_node_idx is before it.start_node_idx
        expire_blocks_before(free_list, it.start);

        // 2) First-fit allocation: Try to find a suitable block in the free_list
        offset = find_first_fit(free_list, it.bytes, alignment=64); [cite: 4, 5]
        if (offset == INVALID) {
            // No fit found, extend the arena from the tail
            offset = align_up(arena_tail, 64); // Ensure 64-byte alignment
            arena_tail = offset + it.bytes;
        } else {
            // Fit found, adjust or remove the free block from the list
            adjust_free_list_entry(free_list, offset, it.bytes);
        }
        plan->offsets[it.value_id] = offset; // Assign the determined offset to the value
    }
    plan->arena_size[arena_class] = arena_tail; // Record the total size needed for this arena
}
```

### 5.4. WORKSPACE Special Rule
The `WORKSPACE` arena is distinct:

*   Its total size is computed as the peak sum of `bytes` for all `WORKSPACE` values that are simultaneously live.
*   Individual `WORKSPACE` values **MAY** be treated as a stack allocator during the linear-scan, meaning their specific offsets are transient during planning. However, for implementation simplicity and to align with the "no allocation in `execute()`" rule, it's often more pragmatic to assign each workspace-using operation its own static slice of the `WORKSPACE` arena, derived from the same linear-scan result.
*   The `plan->arena_size[WORKSPACE]` **MUST** be set to this computed maximum peak.

### 5.5. Output of Planning
Upon successful planning, `vsla_plan_t` will contain:

*   `plan->arena_size[arena_class]` for all arena classes, indicating the total bytes needed.
*   `value[i].arena_offset` for every value `i` in the program, specifying its byte offset within its assigned arena.
*   `plan->schedule`, a list of `vsla_segment_t` objects, defining the execution order and fused kernel groups.

## 6. Arenas
Arenas are pre-allocated, contiguous memory blocks that house tensors during execution. This model prevents runtime allocations in the hot path.

### 6.1. Arena Classes
*   **`VSLA_ARENA_PARAMS`**: For model weights, biases, and other static parameters. These are typically read-only during the forward pass and updated by optimizers after the backward pass.
*   **`VSLA_ARENA_ACTIVATIONS`**: For intermediate tensor outputs from forward pass operations. These may be retained for the backward pass (if not checkpointed).
*   **`VSLA_ARENA_WORKSPACE`**: For transient scratch memory needed by operations (e.g., FFT buffers, temporary sums). Memory in this arena is quickly reused.
*   **`VSLA_ARENA_GRADS`**: For storing gradients with respect to parameters and activations during the backward pass.
*   **`VSLA_ARENA_EXTERNAL`**: For memory buffers managed outside the VSLA system (e.g., user input/output buffers). These are bound to the program at execution time.

### 6.2. Alignment Policy
All memory offsets within an arena **MUST** be aligned to at least 64 bytes. This ensures optimal performance for SIMD operations and cache line utilization.

### 6.3. Profiles (Bucketing)
To avoid dynamic re-planning and re-allocation for variable-shape inputs, VSLA uses profiles:

*   A `vsla_profile_t` represents a pre-compiled (Program, Plan, Executor) tuple for a specific shape family (e.g., `batch_size=32`, `sequence_length=1024`).
*   Programs **MUST** be compiled and planned once per profile. At runtime, the input batch is routed to the appropriate pre-computed profile, enabling execution without any further memory allocations or re-planning.

### 6.4. vsla_shrink Policy
*   Kernels **MUST NOT** automatically perform `vsla_shrink` to their minimal representative. This prevents quadratic complexity cascades (`O(product(shape)*rank)`) that would arise from repeated shrinking operations.
*   `vsla_shrink()` **MAY** be run offline (e.g., during serialization) or explicitly invoked by the caller to convert a tensor to its minimal representative, primarily for storage efficiency.

### 6.5. `vsla_shrink` Policy and Minimal Representatives

The concept of a minimal representative is central to VSLA's efficiency, but its misuse can lead to performance bottlenecks. This section provides clear guidelines on how to handle `vsla_shrink`.

*   **What `vsla_shrink` Does**: The `vsla_shrink` operation reallocates a tensor to its minimal representative, removing any trailing zero hyperplanes. This is useful for serialization or when memory needs to be reclaimed, but it is a potentially expensive operation.

*   **The Dangers of Automatic Shrinking**: If `vsla_shrink` is performed automatically after every operation, it can lead to a quadratic complexity cascade. For example, if you are building up a tensor by repeatedly appending a small chunk, each append operation would trigger a full reallocation and copy of the entire tensor. This would be highly inefficient.

*   **When to Use `vsla_shrink`**: `vsla_shrink` should be used sparingly and only when you are sure that the memory savings are worth the cost of the reallocation. Good candidates for `vsla_shrink` are tensors that have a large number of trailing zeros and are not expected to grow any further.

*   **When to Avoid `vsla_shrink`**: Avoid using `vsla_shrink` in hot loops or on tensors that are frequently modified. In these cases, the cost of the repeated reallocations will likely outweigh any memory savings.

*   **Serialization**: `vsla_shrink` is particularly useful when serializing a tensor. By shrinking the tensor to its minimal representative before serialization, you can significantly reduce the amount of data that needs to be written to disk.

## 7. Execution (CPU Reference)
The Executor is responsible for running the program on the target backend according to the plan. This section outlines the CPU reference implementation.

### 7.1. Segment Representation
The execution plan (`vsla_schedule_t` within `vsla_plan_t`) consists of a sequence of segments. Each segment represents a fused computational block.

```c
typedef vsla_error_t (*vsla_segment_fn)(const vsla_plan_t* plan,
                                      const vsla_value_t* values,
                                      const vsla_node_t* nodes,
                                      size_t first_node, size_t last_node, // Inclusive range of nodes for this segment
                                      const vsla_bindings_t* bind,       // Arena pointers
                                      vsla_stream_t* stream);            // Opaque stream handle for device/threading

typedef struct {
    size_t first_node, last_node; // Inclusive range of IR nodes covered by this segment
    vsla_segment_fn fn;           // Pointer to the specialized (fused) kernel function
} vsla_segment_t; [cite: 4, 5]

typedef struct {
    vsla_segment_t* segments;
    size_t          nsegments;
} vsla_schedule_t; // Embedded inside vsla_plan_t
```

### 7.2. CPU `execute()` Outline
```c
vsla_error_t vsla_program_execute(const vsla_program_t* P,
                                const vsla_plan_t* plan,
                                const vsla_bindings_t* bind, // Contains pointers to pre-allocated arenas
                                const vsla_io_t* in,         // Pointers to program input data (external)
                                vsla_io_t* out,              // Pointers to program output data (external)
                                vsla_stream_t* stream) {     // Opaque stream handle

    // 0) Map input/output host pointers into arena views if needed (for VSLA_ARENA_EXTERNAL)
    // This step ensures that the 'bind' structure correctly points to the external memory for program I/O.
    // No allocation should occur here.
    bind_program_ios(P, plan, bind, in, out);

    // 1) Iterate through each fused segment in the plan's schedule
    for (s = 0; s < plan->schedule.nsegments; ++s) {
        auto seg = plan->schedule.segments[s];
        // Call the specialized kernel function for this segment
        vsla_error_t err = seg.fn(plan, P->values, P->nodes, seg.first_node, seg.last_node, bind, stream); [cite: 4, 5]
        if (err != VSLA_ERR_OK) {
            return err; // Propagate error
        }
    }
    return VSLA_ERR_OK;
}
```

### 7.3. Fusion Rules
Fusion aims to reduce overhead by combining multiple IR nodes into a single, optimized kernel.

**MUST** fuse consecutive nodes if **ALL** the following conditions are true:

*   They operate on the same ambient output shape.
*   All involved operations are element-wise or simple reductions (e.g., add, subtract, Hadamard product, exponential, logarithm, mask, scalar multiplication, residual add).
*   No operation within the chain requires a different arena class or a separate workspace buffer that would break contiguity.
*   No operation acts as an implicit barrier (e.g., FFT, matrix multiplication, convolution, or other complex operations that cannot be easily inlined).

**Example fused element-wise kernel template**:
```c
static vsla_error_t seg_elemwise_chain(const vsla_plan_t* plan,
                                     const vsla_value_t* values,
                                     const vsla_node_t* nodes,
                                     size_t first_node, size_t last_node,
                                     const vsla_bindings_t* bind,
                                     vsla_stream_t* stream){
    // Resolve base pointers for inputs and outputs from the 'bind' structure and 'plan->offsets'.
    float* a = arena_ptr(bind, values[a_id].arena_class, values[a_id].arena_offset);
    float* b = arena_ptr(bind, values[b_id].arena_class, values[b_id].arena_offset);
    float* y = arena_ptr(bind, values[y_id].arena_class, values[y_id].arena_offset);

    size_t N = vsla_logical_elems(out_value_tensor); // Use the ambient size for the loop.

    #pragma omp parallel for schedule(static) // Example threading using OpenMP
    for (size_t i = 0; i < N; ++i) {
        // Operations are directly inlined based on the fused nodes.
        float tmp = a[i] + b[i];       // Corresponds to node 'first_node'
        tmp = tmp * scale;             // Corresponds to node 'first_node + 1'
        y[i] = relu(tmp);              // Corresponds to node 'last_node'
    }
    return VSLA_ERR_OK;
} [cite: 4, 5]
```

### 7.4. Streams
`vsla_stream_t` is an opaque handle representing an execution stream (e.g., a CPU thread-pool queue or a GPU device stream). All operations within a stream **SHOULD** maintain their order and execute asynchronously if supported by the backend, allowing for potential overlap of computation and data transfers.

## 8. Autograd (Reverse-Mode AD)
VSLA supports reverse-mode automatic differentiation through the construction and execution of a backward program (Pᵀ) that computes gradients.

### 8.1. Build Backward Program Pᵀ
The backward program is constructed by traversing the forward program's nodes in reverse topological order. For each node, the Vector-Jacobian Product (VJP) rule is applied to propagate gradients backward.

```c
// Conceptual algorithm:
// grad_map: a mapping from forward value ID to its corresponding gradient value ID in the backward program.
//           Initialized with zero-tensors (buffers) for values that will receive gradients.

for (node in reverse_topo_order(P.nodes)) { [cite: 4, 5]
    // g_outputs are the incoming gradients (cotangent vectors) for the output(s) of the forward node.
    // These are retrieved from `grad_map` based on the forward node's outputs.
    g_outputs = get_gradients_for_outputs(node.outputs, grad_map);

    // emit_vjp(node, g_outputs) generates the necessary backward nodes (ops) for this node,
    // which compute the gradients w.r.t. its inputs based on 'g_outputs'.
    // It returns a tuple of gradient values for each input of the forward node.
    vjp_gradients_for_inputs = emit_vjp_nodes(node, g_outputs); [cite: 4, 5]

    // Accumulate the computed VJP gradients into the respective input gradients in `grad_map`.
    // This handles multiple paths contributing to an input's gradient.
    for each input 'i' of 'node':
        accumulate_gradient(grad_map[i], vjp_gradients_for_inputs[i]); [cite: 4, 5]
}
```
*   `grad_map[x]` (for an input `x`) is initialized to zero-tensors (pre-allocated by the planner) or lazily materialized as needed to ensure a buffer for accumulation.
*   Parameter gradients will accumulate into allocated buffers in the `GRADS` arena.

### 8.2. Checkpointing
Checkpointing is a memory-saving technique for autograd:

*   Specific forward nodes can be explicitly marked to store their outputs (these values are allocated in the `ACTIVATIONS` arena and retained).
*   Any forward value not checkpointed will be recomputed during the backward pass if needed, by emitting the necessary forward subgraph nodes into the backward program (Pᵀ).
*   The planner for the backward pass (`Pb`) **MUST** account for these retained forward activations, ensuring their memory is correctly managed within the `ACTIVATIONS` arena.

### 8.3. Planning & Execution for Backward Pass
*   A separate `vsla_program_t` (B) is built for the backward pass (Pᵀ).
*   A `vsla_plan_t` (Pb) is generated for B. This plan **SHOULD** share the `PARAMS` arena with the forward plan (Pf) if gradients are being accumulated in-place or streamed. It **MAY** optionally share the `ACTIVATIONS` arena if retained activations are passed directly.
*   The backward program is executed after the forward pass, typically provided with the initial `loss_grads` (gradient of the loss with respect to the program's outputs).

### 8.4. Vector-Jacobian Product (VJP) Table
This table defines the backward pass for each forward operation. `g_out` denotes the gradient flowing into the op's output(s). We accumulate into `g_X` for each input `X`. `⊙` denotes Hadamard (element-wise) product.

| Operation | Forward (Y = f(X)) | Gradient w.r.t. Inputs (g_X) | Notes |
| :--- | :--- | :--- | :--- |
| **Elementwise Core** | | | |
| Add | `Y=A+B` | `gA += g_out`, `gB += g_out` | Automatic shape promotion implies padding during forward, but `unprom` (slice/extract) for gradient ensures it flows back to original dimensions. |
| Sub | `Y=A−B` | `gA += g_out`, `gB -= g_out` | |
| Hadamard | `Y=A ⊙ B` | `gA += g_out ⊙ B`, `gB += g_out ⊙ A` | |
| Mul Scalar | `Y=s ⋅ X` | `gX += s ⊙ g_out`, `gs += <g_out, X>` (reduce to scalar) | The inner product `<g_out, X>` implies a sum reduction over all elements to produce the scalar gradient `gs`. |
| Exp | `Y= exp(X)` | `gX += g_out ⊙ Y` | Uses the forward output `Y` (which is `exp(X)`) for efficiency. |
| Log | `Y= log(X)` | `gX += g_out ⊙ (1/X)` | |
| ReLU | `Y= max(0,X)` | `gX += g_out ⊙ 1_{X>0}` | `1_{X>0}` is the indicator function: 1 if `X > 0`, else 0. For `X=0`, gradient is typically 0. |
| Sigmoid | `Y= sigma(X)` | `gX += g_out ⊙ Y ⊙ (1 − Y)` | Uses the forward output `Y` for efficiency. |
| Tanh | `Y= tanh(X)` | `gX += g_out ⊙ (1 − Y²)` | Uses the forward output `Y` for efficiency. |
| GELU | `Y= GELU(X)` | `gX += g_out ⊙ GELU'(X)` | Requires implementation of the GELU derivative (`GELU'(X)`), which depends on the specific GELU approximation (e.g., `tanh` or `erf` variant). |
| Mask | `Y=X ⊙ M` (M is constant) | `gX += g_out ⊙ M` | The mask `M` is not a trainable parameter, so no gradient w.r.t. `M`. |
| Dropout | `Y=X ⊙ M/(1−p)` (M is mask, p is dropout prob) | `gX += g_out ⊙ M / (1-p)` | The random mask `M` (binary, 0 or 1) **MUST** be cached during the forward pass and reused for the backward pass. |
| **Reductions / Broadcast** | | | |
| Sum (axis=K) | `Y= sum_K X` | `gX += broadcast_to_shape(g_out, X.shape)` | The gradient `g_out` is broadcast back to the shape of `X` along the summed axes. |
| Mean (axis=K) | `Y=(1/N) sum_K X` (N is size of K axis) | `gX += broadcast_to_shape(g_out) / N` | Gradient is broadcast and scaled by `1/N`. |
| **Affine & Norms** | | | |
| LayerNorm | `Y=(X− mu)/ sigma ⋅ gamma+ beta` | Standard LayerNorm backward using saved `mu`, `sigma` or recompute. Outputs: `gX`, `gγ`, `gβ`. | |
| RMSNorm | `Y=X/ rms(X) ⋅ gamma` | `gγ += ∑ (g_out ⊙ X / rms(X))` , `gX += …` (complex formula) | The `rms(X)` (root mean square) is `sqrt(mean(X^2))`. The full `gX` derivation involves the chain rule for `1/rms(X)`. |
| **Linear Algebra** | | | |
| Matmul | `Y=A ⋅ B` | `gA += g_out ⋅ B^T`, `gB += A^T ⋅ g_out` | Standard matrix multiplication gradients. |
| Softmax (row-wise) | `Y= softmax(X)` | `gX = g_out ⊙ Y - Y ⊙ ∑_{row}(g_out ⊙ Y)` (per-row reduction) | For `Y_i = exp(X_i) / ∑_j exp(X_j)`. |
| **VSLA Semiring Ops** | | | |
| Conv1D (Model A) | `Y=A ∗conv∗ B` | `gA += g_out *corr* flip(B)`, `gB += flip(A) *corr* g_out` | Correlation is convolution with one operand flipped. FFT path **MAY** be used for efficiency. `flip(X)` reverses the vector. |
| Kronecker (Model B) | `Y= kron(A,B)` | For 1D (vectors) A (length m), B (length n): Reshape `g_out` to (m, n) blocks. Then: `gA[i] += ∑_j g_out[i,j] ⋅ B[j]`
`gB[j] += ∑_i A[i] ⋅ g_out[i,j]` | This is equivalent to summing the outer product of `g_out` with `B` for `gA`, and `A` with `g_out` for `gB`. More generally, for tensors, it involves distributing the gradient to the correct blocks and summing. |
| **Structural** | | | |
| Stack | `Y= Stack(X_1, ...,X_k)` | `gX_i += Slice(g_out, i)` | The `Slice` operation extracts the i-th tensor along the stacking dimension from `g_out` and unpromotes it to `X_i`'s original shape. |
| Window | `Y= Stack(last w inputs)` | Propagate slices back to each contributing input from the window. | This requires tracking which input tensors contributed to each window. |
| Reshape/Transpose/Slice/Gather | Shape-only transforms | `gX = inverse_transform(g_out)` | These operations primarily affect shape metadata. The backward pass involves applying the inverse shape transformation to the incoming gradient. |
| Gather | `Y=X[ indices]` | `gX += scatter(g_out, indices)` (else 0) | `gX` is a sparse tensor where gradients `g_out` are scattered back to their original positions defined by `indices` ; other positions are zero. |

**Implementation Notes for VJP Table**:
*   You **MUST** implement reduction operations for scalar gradients (e.g., `gs` for `MulScalar`).
*   For broadcast/ambient promotion gradients, you **MUST** sum-reduce over the broadcasted axes to correctly accumulate gradients back to the original input shape.

## 9. Cross-Program / Cross-Arena Composition
VSLA programs **MUST** be able to compose and share data effectively, especially for complex models or distributed systems.

### 9.1. Import/Export Values (`vsla_link_t`)
```c
typedef struct {
    const vsla_program_t* src;        // Source program from which a value is exported
    uint32_t              src_value_id; // ID of the value in the source program
    vsla_program_t*       dst;        // Destination program to which the value is imported
    uint32_t              dst_value_id; // ID of the value in the destination program
} vsla_link_t; [cite: 5]
```
*   When a `vsla_link_t` is established, the `dst_value_id` in the destination program gets `arena_class = VSLA_ARENA_EXTERNAL`.
*   The actual memory pointer and offset for `VSLA_ARENA_EXTERNAL` values **MUST** be supplied at bind time (`vsla_program_bind`).
*   The caller is responsible for enforcing correct sequencing of program executions (e.g., source program completes before destination program starts) or using backend-specific stream/event synchronization if supported.

### 9.2. Shared Arena Manager
A higher-level Shared Arena Manager **MAY** be implemented. This manager would pre-allocate a large, unified memory region (a "mega-arena") for all programs.

In such a setup, individual program plans would be colored in a global pass by the manager, ensuring all offsets are relative to the mega-arena. This allows seamless data sharing and reduces fragmentation across multiple programs.

## 10. Backend Abstraction (C vtable)
VSLA's architecture supports multiple hardware backends (CPU, GPU, etc.) via a C vtable, enabling device-agnostic program definition.

```c
typedef struct vsla_backend_vtable_s {
    // Memory management functions
    void* (*alloc)(size_t bytes, size_t alignment, void* ctx);           // Allocates device memory
    void  (*free)(void* ptr, void* ctx);                                 // Frees device memory
    void  (*memcpy_h2d)(void* dst, const void* src, size_t bytes, void* ctx); // Host to device memory copy
    void  (*memcpy_d2h)(void* dst, const void* src, size_t bytes, void* ctx); // Device to host memory copy
    void  (*memset_zero)(void* dst, size_t bytes, void* ctx);            // Fills device memory with zeros

    // Execution functions
    vsla_error_t (*launch_segment)(const vsla_segment_t* segment,
                                 const vsla_plan_t* plan,
                                 vsla_stream_t* stream,
                                 void* ctx);                             // Launches a fused segment on device
    vsla_error_t (*synchronize)(vsla_stream_t* stream, void* ctx);        // Synchronizes the stream

    // Optional graph recording/replay functions (e.g., CUDA Graphs)
    vsla_error_t (*record_graph)(void* program_handle, vsla_stream_t* stream, void* ctx, void** graph_handle_out); // Records execution graph
    vsla_error_t (*replay_graph)(void* graph_handle, vsla_stream_t* stream, void* ctx); // Replays recorded graph
} vsla_backend_vtable_t; [cite: 5]

typedef struct {
    const vsla_backend_vtable_t* vtbl; // Pointer to the backend's function table
    void*                        ctx;  // Backend-specific context handle (e.g., CUDA device ID)
} vsla_backend_t; [cite: 5]
```
A CPU backend **MUST** provide a correct and simple baseline implementation for all required functions.

## 11. Public C API (End-to-End Lifecycle)
This section illustrates the typical lifecycle of a VSLA computation using the public C API.

### 11.1. Compile (Forward Pass)
```c
// 1. Build the program (IR) from a user-defined graph description.
vsla_program_t* P = vsla_program_build(&graph_description, &build_options);
if (!P) { /* Handle error from vsla_get_last_error() */ }

// 2. Plan the program for a specific backend (memory layout, execution schedule).
vsla_plan_t* Pf = vsla_program_plan(P, &plan_options, backend);
if (!Pf) { /* Handle error */ vsla_program_destroy(P); }

// 3. Bind the plan to allocate arenas and set up value offsets.
vsla_bindings_t bf = {0}; // Structure to hold arena pointers
vsla_error_t err = vsla_program_bind(P, Pf, backend, &bf);
if (err != VSLA_ERR_OK) { /* Handle error */ vsla_plan_destroy(Pf); vsla_program_destroy(P); }
// At this point, all necessary memory for the forward pass is allocated and mapped.
```

### 11.2. Execute (Forward Pass)
```c
// Prepare input data (e.g., copy from host to device if not CPU backend).
vsla_io_t inputs = { /* ... populate with input tensor data */ };
vsla_io_t outputs = { /* ... populate with output tensor placeholders */ };
vsla_stream_t* stream = vsla_backend_create_stream(backend); // Create an execution stream

// Execute the forward pass. This call MUST NOT allocate memory.
vsla_error_t err = vsla_program_execute(P, Pf, &bf, &inputs, &outputs, stream);
if (err != VSLA_ERR_OK) { /* Handle error */ }

vsla_backend_synchronize(stream, backend->ctx); // Ensure all operations complete
// Results are now in 'outputs' buffers.

// Release the stream (if applicable)
vsla_backend_destroy_stream(stream);
```

### 11.3. Compile & Run (Backward Pass)
```c
// 1. Build the backward program (Pᵀ) from the forward program.
// This involves generating VJP nodes based on the forward graph.
vsla_program_t* B = vsla_autograd_build_backward(P);
if (!B) { /* Handle error */ }

// 2. Plan the backward program. This plan MAY share PARAMS and ACTIVATIONS arenas with Pf.
vsla_plan_t* Pb = vsla_autograd_plan_backward(B, Pf, &ad_options, backend);
if (!Pb) { /* Handle error */ vsla_program_destroy(B); }

// 3. Bind the backward plan to allocate its arenas (e.g., GRADS arena)
// and set up value offsets.
vsla_bindings_t bb = {0};
vsla_error_t err_b = vsla_program_bind(B, Pb, backend, &bb);
if (err_b != VSLA_ERR_OK) { /* Handle error */ }
// Note: bb.params_arena_ptr MAY be set to bf.params_arena_ptr for shared parameter storage.

// Prepare initial loss gradients and storage for parameter gradients.
vsla_io_t loss_grads = { /* ... populate with loss gradient data */ };
vsla_io_t param_grads = { /* ... populate with parameter gradient placeholders */ };
vsla_stream_t* b_stream = vsla_backend_create_stream(backend);

// Execute the backward pass. This call MUST NOT allocate memory.
err_b = vsla_autograd_execute(B, Pb, &bb, &inputs, &outputs, &loss_grads, &param_grads, b_stream);
if (err_b != VSLA_ERR_OK) { /* Handle error */ }

vsla_backend_synchronize(b_stream, backend->ctx); // Synchronize backward stream
// Parameter gradients are now available in 'param_grads'.

// Clean up
vsla_backend_destroy_stream(b_stream);
vsla_program_destroy(B);
vsla_plan_destroy(Pb);
```

### 11.4. Profiles Usage
```c
// 1. Compile a profile for a specific shape bucket (e.g., seq_len = 2048).
// This internally builds the program, plans it, and binds arenas.
vsla_profile_t* prof = vsla_profile_compile(&graph_description, &bucket_2048_config, backend);
if (!prof) { /* Handle error */ }

// 2. Execute using the profile. This call is highly optimized and MUST NOT re-allocate.
// The profile manages its internal program, plan, and bindings.
vsla_error_t err_prof = vsla_profile_run(prof, &inputs, &outputs, stream);
if (err_prof != VSLA_ERR_OK) { /* Handle error */ }

// The profile can be run repeatedly for any inputs matching 'bucket_2048_config'.
// Clean up the profile when no longer needed.
vsla_profile_destroy(prof);
```

## 12. Transformer Worked Example (Arena Sizing Formulas, Profiles)
This example illustrates memory requirements for a Transformer model. Assume a model with `L` layers, `d_model` embedding dimension, `h` attention heads, `d_head` head dimension, `d_ff` feed-forward dimension, `V` vocabulary size, `seq_len` bucket `S`, and `batch_size` `B`.

### 12.1. Parameter Memory (`VSLA_ARENA_PARAMS`)
Total parameters (`P`) = sum of:
*   **Embeddings**: `V * d_model`
*   **QKV projections** (per layer): `d_model * 3 * d_model` (for Q, K, V linear layers)
*   **Attention output** (per layer): `d_model * d_model`
*   **Feed-Forward** (2 linear layers per layer): `d_model * d_ff + d_ff * d_model`
*   **LayerNorm/RMSNorm** (per layer): `2 * d_model` (for `γ, β` if LayerNorm, or `d_model` if RMSNorm for `γ` only)

Total `PARAMS` bytes = `P * sizeof(dtype)`. This arena is shared across all sequence length buckets.

### 12.2. Activation Memory (`VSLA_ARENA_ACTIVATIONS`)
Rough upper bound for the forward pass (without checkpointing):
*   **Token embeddings**: `B * S * d_model`
*   **Q, K, V tensors**: `3 * B * S * d_model`
*   **Attention scores**: `B * h * S * S` (for dense full attention)
*   **Attention probabilities**: `B * h * S * S`
*   **Attention output**: `B * S * d_model`
*   **MLP intermediates**: `B * S * d_ff`
*   **Residuals/norms**: roughly `O(B * S * d_model)` per layer (these can often be overlapped/reused by the planner).

The planner will significantly reduce the total `ACTIVATIONS` memory via liveness coloring, by re-using memory for tensors that are no longer needed.

### 12.3. Workspace Memory (`VSLA_ARENA_WORKSPACE`)
This arena holds transient buffers.
*   FFT buffers (if convolution operations are used).
*   Softmax temporary reductions.

`WORKSPACE` bytes = Maximum over all nodes of (workspace needed while that node's live window intersects others). The linear-scan memory planner will compute this peak usage.

### 12.4. Gradients Memory (`VSLA_ARENA_GRADS`)
*   **`GRADS` params**: Typically the same size as `PARAMS` arena (unless gradients are streamed and updated in-place without full materialization).
*   **`GRADS` activations**: Similar magnitude to forward activations, unless aggressive checkpointing is used to recompute rather than store them.

### 12.5. Profiles
*   Generate (`Program`, `Plan`) for each sequence length `S` in a defined set (e.g., `{512, 1024, 2048, 4096}`).
*   Store `arena_size[class]` for each bucket. At runtime, inputs are routed to the appropriate `S` bucket, and the pre-computed plan is bound and executed.

**Example Memory Report (for S=2048)**:
*   **`PARAMS`**: X GiB (shared across all buckets)
*   **`ACTIVATIONS` (fwd)**: Y GiB
*   **`WORKSPACE`**: Z MiB
*   **`GRADS`**: G GiB

## 13. Numerics & Precision Policy
*   **Accumulation**: All reductions and FFT temporaries **MUST** use `double` precision for intermediate accumulations to maintain numerical stability.
*   **Storage**: Tensor elements **MAY** be stored in `float32`, `float16`, or `bfloat16`. Downcasting from `double` occurs upon storing results.
*   **FMA**: Use `fma` (fused multiply-add) instructions where available (e.g., convolution inner loops) for improved precision and performance.
*   **Comparison Tolerance**: For floating-point comparisons in tests, use `abs(x-y) ≤ atol + rtol*max(|x|,|y|)`. Recommended default tolerances: `(atol, rtol) = (1e-12, 1e-9)` for `double` and relaxed values for `float` (e.g., `1e-6, 1e-3` for `fp32`).
*   **FFT Error**: Forward/inverse FFT operations have an error bound of `O(ε log L)` (where `ε` is machine epsilon). The inverse FFT **MUST** scale its output by `1/L`. Double temporaries help reduce cancellation errors.
*   **Flush-to-zero**: An optional global toggle **MAY** be provided to flush denormal numbers to zero for performance, but this can impact numerical exactness.
*   **Empty Tensors**: All kernels **MUST** guard against `data==NULL` dereferences and early-exit gracefully when encountering empty tensors (logical size = 0).

## 14. Testing Matrix
Comprehensive testing is crucial for ensuring correctness and performance across all VSLA components.

| Category | MUST Pass Checks |
| :--- | :--- |
| **IR Build & Validation** | Topological sort succeeds; invalid dtypes are rejected; shapes are correctly inferred for all ops; `capacity >= shape` is enforced. |
| **Planning** | No overlapping memory intervals within an arena; all offsets are 64-byte aligned; arena sizes are stable across identical planning runs. |
| **Determinism** | N identical runs produce bit-for-bit identical results; zero dynamic allocations are observed during `execute()` (verify with `malloc`/`free` overrides). |
| **Autograd Core** | Finite-difference gradient checks on random graphs and shapes: maximum relative error **MUST** be below the defined tolerance. |
| **Profiles** | Selecting different buckets **MUST NOT** trigger re-planning or re-allocation. |
| **Backends** | CPU vs. GPU numerical parity **MUST** be within the defined tolerance. |
| **Cross-Program** | `vsla_link_t` import/export mechanisms work correctly; external arena offsets are valid. |
| **Fuzz Testing** | Randomly generated DAGs (within bounded size), shapes, and dtypes are processed correctly for `10^4` or more cases. |
| **Unit Tests** | * **Addition/Subtraction**: `[1,2,3] + [4,5] → [5,7,3]`; subtraction producing trailing zeros correctly shrinks.<br> * **Hadamard**: Correct element-wise product.<br> * **Kronecker**: Non-commutativity `kron([1,2],[0,1])` vs. reversed is verified.<br> * **Convolution**: Direct vs. FFT path equivalence within tolerance for sizes above threshold; FFT plan cache reuse; empty operands; shrink correctness.<br> * **Stacking**: Heterogeneous shapes stack correctly; ambient promotion is verified.<br> * **Window Pyramid**: Feed `2^L` tensors; verify cascade height and outputs.<br> * **Empty Operands**: All operations handle empty operands gracefully and produce correct empty/zero results.<br> * **FFT Specific**: Plan cache reuse; numerical accuracy; edge cases for `L` (e.g., 1, 2, 4). |
| **Property-Based Testing** | Associativity (addition, Kronecker); Distributivity; Scalar identity; `(a⊗b)⊗c = a⊗(b⊗c)`. |
| **Property-Based Testing** | In addition to the existing property tests, the following should be added:<br> * **Idempotence**: `shrink(shrink(T)) == shrink(T)`.<br> * **Equivalence**: `(T + Z) == T` where `Z` is a zero tensor.<br> * **Shape Invariance**: `shape(T + Z) == shape(T)`.<br> * **Rank Invariance**: `rank(T + Z) == rank(T)`.<br> * **Preservation of Trailing Zeros**: `(T + Z)` has the same number of trailing zeros as `T`. |

## 15. Performance Checklist
These are critical guidelines for achieving high performance in a VSLA implementation.

*   **Fuse Elementwise Chains**: Implement segment fusion for element-wise operations to reduce kernel launch overhead.
*   **64-byte Alignment**: All memory allocations and tensor offsets **MUST** adhere to 64-byte alignment to enable optimal SIMD and cache performance.
*   **Cache FFT Plans**: FFT plans (twiddles, bit-reversal tables) **MUST** be cached by FFT length (`L`) to avoid redundant computation.
*   **Persistent Kernels / CUDA Graphs**: For GPU backends, leverage persistent kernels or CUDA Graphs to minimize driver overhead for repeated executions.
*   **NUMA-aware Partitions**: (Optional) For large-scale CPU deployments, consider NUMA-aware memory placement strategies.
*   **Profile-Guided Specialization**: The planner **MAY** use profiling information to specialize the plan (e.g., grouping hot tensors for better cache locality).
*   **No Dynamic Allocations in `execute()`**: This is a fundamental invariant. Ensure that runtime execution paths perform no `malloc` or `free` calls.

## 16. Migration Guide (v3.2 → v4.3)
This guide outlines the transition process from the v3.2 API to the v4.3 program/plan/execute architecture.

*   **Keep v3.2 Primitive Kernels**: The underlying arithmetic (`vsla_add`, `vsla_conv`, `vsla_kron`, etc.) from v3.2 can be largely retained. These will now be called from within the new `vsla_segment_fn` implementations defined by the executor.
*   **Wrap Legacy APIs**: The old per-operation APIs (`vsla_add(out, a, b)`) **SHOULD** be wrapped as single-operation `vsla_program_t` objects for backward compatibility. This allows existing v3.2 code to gradually migrate.
*   **Incremental Autograd**: Implement the autograd VJP table incrementally, starting with core operations, using the detailed specifications in Section 8.4.
*   **Adopt New Codepaths**: New development **SHOULD** exclusively use the program, plan, and execute interfaces to leverage the architectural benefits.
*   **Memory Model Transition**: Understand that `vsla_tensor_t`s no longer manage their own memory independently via `refcnt` (though `refcnt` might still be used for debug or external ownership). Memory ownership shifts to `vsla_plan_t` and `vsla_bindings_t`. Manual `vsla_retain`/`vsla_release` on tensors will largely be replaced by the planner's allocation/deallocation of arenas.

## 17. Future Work
*   **Dynamic Shapes**: Support dynamic shapes within a single `vsla_program_t` through guarded sub-allocators or dynamic re-planning on shape changes (more sophisticated than current bucketing).
*   **Operator Autotuning**: Implement operator autotuning mechanisms (e.g., for FFT thresholds, tiling strategies, block sizes) to adapt to specific hardware and input characteristics.
*   **Compiler Passes**: Develop IR-level compiler passes for optimizations such as Common Subexpression Elimination (CSE), Dead Code Elimination (DCE), and layout optimization.
*   **Distributed Systems**: Extend arena and execution models to support distributed/multi-device computing environments.
*   **Higher-Level Language Bindings**: Build robust higher-level language bindings (e.g., Python, Rust) atop the stable C ABI.
*   **Automated Broadcast-Reduction Gradient Generation**: Develop a mechanism to automatically generate broadcast/reduction gradients in the VJP rules to reduce boilerplate and potential errors.
*   **Higher-Rank Convolution**: Progress the implementation and detailed specification for axis-wise higher-rank convolution operations.
*   **Sparse Extension (Interior Zeros)**: Investigate and formalize mechanisms for handling interior (non-trailing) zeros efficiently within the VSLA framework, potentially via masks or specialized sparse formats.
*   **Categorical Formulation**: A complete categorical treatment of VSLA as a semiring-enriched category could provide deeper structural insights and new optimization opportunities.
*   **Topological Considerations**: Formalizing the preservation of topological properties (e.g., connectivity, causality) could enable certified correctness for safety-critical applications.
*   **Information-Theoretic Analysis**: Investigate fundamental limits on compression through variable-shape representations and the relationship between shape entropy and computational complexity.
*   **Synergies with Semi-Tensor Product (STP)**: Explore hybrid algebraic models where VSLA's equivalence classes integrate with STP's logical and control-theoretic tools.
*   **Quantum Computing Applications**: Investigate VSLA's suitability for quantum simulation, hybrid quantum-classical optimization, and efficient simulation of quantum error correction codes due to its variable-dimensional nature.
*   **Edge Computing and Distributed Systems**: Develop ultra-low-power implementations and federated learning protocols leveraging VSLA's memory efficiency for resource-constrained environments.
*   **Domain-Specific Applications**: Continue exploring applications in computational biology, climate modeling, and financial engineering, adapting VSLA to their unique data structures and computational needs.

End of v4.3 Document. This guide provides the authoritative specification for the VSLA C implementation. Adherence to its norms, especially the "**MUST**" and "**SHOULD**" clauses, is critical for achieving correct, high-performance, and maintainable code.