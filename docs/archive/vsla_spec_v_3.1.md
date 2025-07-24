# VSLA: C Implementation Guide v3.1

**Objective.** Provide a *mathematically rigorous* and *implementation‑ready* specification for the VSLA C backend. This document is normative: coding agents **must** adhere to it unless an explicit deviation note accompanies the change.

---

## Table of Contents

1. Mathematical Model\
   1.1 Terminology\
   1.2 Equivalence Classes & Minimal Representatives\
   1.3 Semiring Models\
   1.4 Special Cases
2. Runtime Data Structures\
   2.1 Ownership & Reference Counting\
   2.2 Invariants\
   2.3 Helper Functions
3. API Contracts\
   3.1 Pre‑Conditions\
   3.2 Post‑Conditions\
   3.3 Error Codes
4. Arithmetic Operators\
   4.1 Elementwise Add/Subtract\
   4.2 Hadamard Product\
   4.3 Convolution (Model A)\
   4.4 Kronecker Product (Model B)
5. Structural Operators\
   5.1 Stacking `S_k`\
   5.2 Window Stacking & Pyramids
6. Shrinking to Minimal Representative
7. Numerical & DType Considerations
8. Parallelization Roadmap
9. Testing Strategy
10. Complexity Summary
11. Coding Agent Checklist
12. Future Work

---

## 1. Mathematical Model

### 1.1 Terminology

- **Rank** `r`: number of axes. `0 ≤ r ≤ VSLA_MAX_RANK`.
- **Tensor:** Any `vsla_tensor_t` with rank `r ≥ 1`. Rank‑0 tensors are avoided to simplify allocation, shape arithmetic, and FFI boundaries. (The struct allows `rank==0` but production code should not construct it.)
- **Vector:** Tensor of rank 1.
- **Scalar:** Canonical representation is a rank‑1 tensor with shape `[1]`. All references to “scalar” below mean shape `[1]`.

### 1.2 Equivalence Classes & Minimal Representatives

Let a tensor of rank `r` have shape `d = (d0,…, d_{r-1})` where each `di ≥ 0`. Its logical element count is `D = ∏ di`; if any `di==0` then `D=0` (empty tensor). Data is stored in row‑major order.

Two shape/data pairs `(d,v)` and `(d',w)` are *equivalent* if, after padding each into the ambient join `n = elementwise_max(d,d')` using **trailing‑zero padding** (inject at the origin; fill uncovered indices with zeros), the padded arrays are bitwise identical. A VSLA tensor denotes such an equivalence class.

**Minimal Representative.** Each class has a unique representative whose shape has no *trailing zero hyperplanes*: scanning axes from last to first, repeatedly decrement the final coordinate while the terminal hyperplane is all zero. Runtime storage must always use minimal representatives unless explicitly stated (e.g., before a deferred `vsla_shrink`).

### 1.3 Semiring Models

Two semirings are exposed over equivalence classes (addition always defined via ambient promotion + pointwise operation):

- **Model A – Convolution Semiring**\
  *Addition/Subtraction:* ambient promotion + elementwise op.\
  *Multiplication:* discrete 1‑D convolution on vectors. Commutative; scalar `[1]` is multiplicative identity.
- **Model B – Kronecker Semiring**\
  *Addition/Subtraction:* as above.\
  *Multiplication:* Kronecker product on vectors. Generally non‑commutative; scalar `[1]` commutes.

### 1.4 Special Cases

- **Scalar:** shape `[1]`; multiplicative identity under both models.
- **Empty tensor:** Any shape with a zero dimension. Logical size = 0. Implementation invariant: if logical *or* capacity product is 0 set `data==NULL` (no allocation). Arithmetic kernels must branch on emptiness and never dereference `data`.
- **Empty operands:** Addition treats empties as zero; convolution/Kronecker with an empty operand yields empty; stacking emits zeros for that slice.

---

## 2. Runtime Data Structures

```c
struct vsla_tensor {
    uint64_t* shape;     // length == rank; minimal logical dimensions
    uint64_t* capacity;  // length == rank; capacity[i] >= shape[i]
    uint8_t    rank;     // 0 <= rank <= VSLA_MAX_RANK
    void*      data;     // contiguous row-major buffer sized product(capacity)
    vsla_model_t model;  // VSLA_MODEL_A or VSLA_MODEL_B
    vsla_dtype_t dtype;  // element type (float32,float64,...)
    _Atomic uint32_t refcnt; // intrusive reference count
    // optional: allocator handle, flags, user data
};
```

### 2.1 Ownership & Reference Counting

```c
void vsla_retain(vsla_tensor_t* t){ if(t) atomic_fetch_add(&t->refcnt,1); }
void vsla_release(vsla_tensor_t* t){
    if(!t) return;
    if(atomic_fetch_sub(&t->refcnt,1)==1){ free(t->shape); free(t->capacity); free(t->data); free(t);} }
```

Ownership contract: functions taking `const vsla_tensor_t*` neither mutate nor retain. Creators return tensors with `refcnt=1`. Functions writing to an existing `out` parameter never change its refcount. Destructors are idempotent under over‑release avoidance.

**Thread‑Local Error State** for constructors returning `vsla_tensor_t*`:

```c
vsla_error_t vsla_get_last_error(void); // read-only; write occurs on failure
```

No global errno usage; avoids interference with external libraries.

### 2.2 Invariants

1. **Capacity dominance:** `capacity[i] ≥ shape[i]`. Policy: `capacity[i] = next_pow2(shape[i])` on creation; shrinking never reduces capacity.
2. **Strides:** Not stored; computed on demand using *capacity* dimensions for O(1) extras. (A future optimization may cache them.)
3. **Allocation size:** `bytes = sizeof(dtype)*product(capacity[i])`. If product is 0 allocate nothing and set `data==NULL`. Overflow detection precedes multiplication.
4. **Slack:** Elements with any index ≥ `shape[i]` are uninitialized and must not be read.
5. **Zero Initialization:** On creation, the logical region is zeroed; slack is unspecified.
6. **Empty tensors:** `data==NULL`.
7. **Minimality:** After `vsla_shrink` (or before serialization), trailing zero hyperplanes must have been removed.

### 2.3 Helper Functions

```c
bool vsla_is_empty(const vsla_tensor_t* t);
uint64_t vsla_logical_elems(const vsla_tensor_t* t);   // product(shape) w/ overflow guard
uint64_t vsla_capacity_elems(const vsla_tensor_t* t);  // product(capacity)
```

Overflow guard:

```c
static inline bool mul_ov(uint64_t a,uint64_t b){ return b && a>UINT64_MAX/b; }
```

Strides / offset (on demand):

```c
static inline void compute_strides(const vsla_tensor_t* t,uint64_t* s){
    uint64_t acc=1; for(int j=t->rank-1;j>=0;--j){ s[j]=acc; acc*=t->capacity[j]; }
}
static inline uint64_t vsla_offset(const vsla_tensor_t* t,const uint64_t* idx){
    uint64_t strides[VSLA_MAX_RANK]; compute_strides(t,strides);
    uint64_t off=0; for(int j=0;j<t->rank;++j) off+=idx[j]*strides[j]; return off; }
```

`unravel` (mixed‑radix):

```c
void unravel(uint64_t lin,const uint64_t* shape,uint8_t rank,uint64_t* out){
    for(int j=rank-1;j>=0;--j){ uint64_t s=shape[j]; out[j]=(s? lin%s:0); lin/=(s? s:1); } // zero-sized dims: out[j]=0 and lin unchanged (division by 1 avoids /0)
}
}
```

---

## 3. API Contracts

Binary op signature:

```c
vsla_error_t vsla_op(vsla_tensor_t* out,
                     const vsla_tensor_t* a,
                     const vsla_tensor_t* b);
```

### 3.1 Pre‑Conditions

1. Non‑NULL pointers (except an empty tensor may have `data==NULL`).
2. `a->model == b->model == out->model`.
3. Identical `dtype` across all operands.
4. Rank compatibility: elementwise ops require `a->rank == b->rank == out->rank`; convolution/Kronecker require rank==1.
5. Output shape must exactly equal the shape rule of the operation. Caller allocates `out` beforehand; no implicit resize.
6. No aliasing: `out` must not alias `a` or `b`. (Future extension may permit controlled aliasing.)
7. All internal shape multiplications pass overflow guards.

### 3.2 Post‑Conditions

On success `out` holds (possibly un‑shrunk) representative of result and all invariants. On failure: `out` is left untouched (implementation may optionally zero its logical region). Error state is reported via return code.

### 3.3 Error Codes

`VSLA_ERR_OK`, `VSLA_ERR_NULL`, `VSLA_ERR_MODEL_MISMATCH`, `VSLA_ERR_RANK`, `VSLA_ERR_SHAPE`, `VSLA_ERR_DTYPE_MISMATCH`, `VSLA_ERR_OVERFLOW`, `VSLA_ERR_ALLOC`, `VSLA_ERR_UNSUPPORTED`.

---

## 4. Arithmetic Operators

### 4.1 Elementwise Addition / Subtraction

Shape rule: `out->shape[i] = max(a->shape[i], b->shape[i])`. Empty operands behave as zeros.

Reference pseudocode:

```c
static inline bool in_bounds(const vsla_tensor_t* t,const uint64_t* idx){
    for(int d=0; d<t->rank; ++d) if(idx[d]>=t->shape[d]) return false; return true; }
size_t total = vsla_logical_elems(out); // ambient size
for(size_t lin=0; lin<total; ++lin){
    uint64_t idx[VSLA_MAX_RANK]; unravel(lin,out->shape,out->rank,idx);
    double va = in_bounds(a,idx)? ((double*)a->data)[vsla_offset(a,idx)]:0.0;
    double vb = in_bounds(b,idx)? ((double*)b->data)[vsla_offset(b,idx)]:0.0;
    ((double*)out->data)[vsla_offset(out,idx)] = va + vb; // or va - vb
}
```

**Complexity:** `∏_i max(d^a_i,d^b_i)`.

**Optimization:** Dominance fast path → if `a` already has ambient shape and `b` is embedded: `memcpy(out,a)` then accumulate inside `b`’s bounding box. Symmetric for `b`.

### 4.2 Hadamard (Elementwise) Product

Same loop, assignment `OUT[...] = va * vb`. Ambient semantics chosen for distributivity. (Implementation may detect non‑overlapping region to early exit with zeros.)

### 4.3 Convolution (Model A)

Rank: vectors only. Let `m=a->shape[0]`, `n=b->shape[0]`. Empty operand ⇒ empty result. Shape rule: `out->shape[0] = (m==0||n==0)?0 : m+n-1`.

**Direct:**

```c
for(uint64_t k=0;k<out_n;++k){
    double sum=0.0;
    uint64_t lo = (k < n-1? 0 : k-(n-1));
    uint64_t hi = (k < m-1? k : m-1);
    for(uint64_t i=lo;i<=hi;++i) sum += A[i]*B[k-i];
    OUT[k]=sum;
}
```

Complexity `O(mn)`.

**FFT Path:** for `mn` exceeding empirical threshold `CONV_FFT_THRESHOLD` (choose where `mn ≈ (m+n) log2(m+n) * c_fft`). Steps: (1) `L = next_pow2(m+n-1)`; (2) allocate complex arrays `fa[L]`, `fb[L]`; (3) forward FFT; (4) pointwise multiply; (5) inverse FFT; (6) copy real parts; (7) (optional) shrink coefficients whose magnitude `< 32 * DBL_EPSILON * max(|out|)`. Double precision temporaries mitigate rounding. Cache FFT plans globally.

### 4.4 Kronecker Product (Model B)

Rank: vectors only. `m=a->shape[0]`, `n=b->shape[0]`. Shape rule: `out->shape[0] = (m==0||n==0)?0 : m*n`.

```c
for(uint64_t i=0;i<m;++i){ double ai=A[i]; double* dst=OUT+i*n; for(uint64_t j=0;j<n;++j) dst[j]=ai*B[j]; }
```

Complexity `O(mn)`. Non‑commutative unless one operand is scalar `[1]`.

---

## 5. Structural Operators

### 5.1 Stacking Operator `S_k`

Inputs: `T0,…,T_{k-1}` each rank `r`. Ambient per‑axis max `A[j]`. Output rank `r+1`, shape `(k, A0,…,A_{r-1})`.

```c
vsla_tensor_t* vsla_stack(const vsla_tensor_t* const* tensors,size_t k);
```

Algorithm: compute ambient shape (guard overflow), allocate output zeroed. For each input copy its logical region into slice `i`. Empty inputs leave slice zero. Optional: shrink along new axis if trailing slices zero.

Complexity: `Σ_i product_j T_i.shape[j]`.

### 5.2 Window Stacking & Pyramids

Maintain ring buffer collecting `w` tensors then emitting `S_w`:

```c
typedef struct { vsla_tensor_t** buf; size_t fill,w; } vsla_window_t;
```

```c
vsla_tensor_t* vsla_window_push(vsla_window_t* W, vsla_tensor_t* x){
    vsla_retain(x); W->buf[W->fill++] = x;
    if(W->fill==W->w){ vsla_tensor_t* out=vsla_stack(W->buf,W->w);
        for(size_t i=0;i<W->w;++i) vsla_release(W->buf[i]); W->fill=0; return out; }
    return NULL; }
```

A pyramid is an array `windows[L]`; feed results recursively upward. Flushing policy (discard or pad partials) is caller‑defined. Amortized linear time.

---

## 6. Shrinking to Minimal Representative

Trailing zero hyperplanes may arise (e.g., after addition). Kernels **must not** perform shrink automatically to avoid quadratic cascades; callers invoke `vsla_shrink` or `vsla_finalize` explicitly.

Algorithm: For each axis from last to first, while the terminal hyperplane is all zeros decrement that dimension. Empty tensors early‑exit. Hyperplane enumeration touches `plane_elems = ∏_{j≠axis} shape[j]` elements.

Pseudocode excerpt:

```c
void vsla_shrink(vsla_tensor_t* t){
    if(!t) return; for(int j=0;j<t->rank;++j) if(t->shape[j]==0) return; // already empty
    uint64_t strides[VSLA_MAX_RANK]; compute_strides(t,strides);
    for(int axis=t->rank-1; axis>=0; --axis){
        while(t->shape[axis] > 0){
            uint64_t last = t->shape[axis]-1; bool all_zero = true; uint64_t plane_elems=1;
            for(int j=0;j<t->rank;++j) if(j!=axis) plane_elems*=t->shape[j];
            uint64_t idx[VSLA_MAX_RANK]={0}; idx[axis]=last;
            for(uint64_t p=0;p<plane_elems && all_zero;++p){
                double val=((double*)t->data)[vsla_offset(t,idx)];
                if(val!=0.0){ all_zero=false; break; }
                for(int j=t->rank-1;j>=0;--j){ if(j==axis) continue; if(++idx[j]<t->shape[j]) break; idx[j]=0; }
            }
            if(!all_zero) break; --t->shape[axis];
        }
    }
}
```

Complexity worst‑case `O(product(shape)*rank)`; typically dominated by shrinkable suffix size.

---

## 7. Numerical & DType Considerations

- **Accumulation:** use double; downcast on store for float storage.
- **FMA:** Use `fma` where available (convolution inner loop) for precision and performance.
- **Comparison tolerance:** `abs(x-y) ≤ atol + rtol*max(|x|,|y|)` with `(atol,rtol)=(1e-12,1e-9)` for double; relax values for float.
- **FFT:** forward/inverse error `O(ε log L)` (ε machine epsilon). Double temporaries reduce cancellation.
- **Flush‑to‑zero:** optional global toggle for denormals.
- **Empty tensors:** guard all dereferences; any kernel encountering empties should early‑exit.

---

## 8. Parallelization Roadmap

Current build is **not** internally thread‑safe. Planned improvements:

- Elementwise ops: partition outermost index ranges.
- Kronecker: parallelize outer loop.
- FFT: parallel batched transforms; share plans across threads.
- Memory: allocate 64‑byte aligned buffers to enable SIMD loads/stores.

---

## 9. Testing Strategy

Deterministic unit tests:

1. Addition `[1,2,3] + [4,5] → [5,7,3]`; subtraction producing trailing zeros → shrink.
2. Kronecker non‑commutativity: `kron([1,2],[0,1])` vs reversed.
3. Convolution: direct vs FFT for sizes above threshold (within tolerance).
4. Stacking heterogeneous shapes; verify ambient promotion.
5. Window pyramid: feed `2^L` tensors; verify cascade height & outputs.
6. Empty operands across all ops.
7. Fuzz: random shapes & models; compare against dense reference (explicit padding) + property tests.

Property tests: associativity (addition, Kronecker), distributivity, scalar identity, `(a⊗b)⊗c = a⊗(b⊗c)`.

---

## 10. Complexity Summary

| Operation            | Rank‑1 Lengths (m,n) | Complexity        | Notes                  |
| -------------------- | -------------------- | ----------------- | ---------------------- |
| Addition/Subtract    | m,n                  | O(max(m,n))       | ambient shape          |
| Hadamard             | m,n                  | O(max(m,n))       | ambient shape          |
| Convolution (direct) | m,n                  | O(m·n)            | threshold‑controlled   |
| Convolution (FFT)    | m,n                  | O((m+n) log(m+n)) | `L = next_pow2(m+n-1)` |
| Kronecker            | m,n                  | O(m·n)            | output length `m·n`    |
| Stack k tensors      | {m\_i}               | Σ m\_i            | zero padding implicit  |
| Window Pyramid       | stream               | Amortized O(N)    | linear in inputs       |

---

## 11. Coding Agent Checklist

1. Compute output shape first; allocate & validate before writing.
2. Never read slack; use bounds or dominance fast path.
3. Early‑exit on empty operands.
4. No in‑place writes unless explicitly supported.
5. Guard every shape multiplication against overflow.
6. Cache FFT plans; do not re‑plan per call.
7. Defer shrink until finalization to avoid quadratic overhead.
8. Respect Model B operand order (non‑commutative).
9. Use `restrict` pointers and alignment hints for performance.
10. On errors: free temporaries, leave `out` untouched, set error code.

---

## 12. Future Work

- Higher‑rank convolution (axis‑wise).
- Batched FFT via stacking; GPU kernels.
- Mixed precision (float storage with double accumulation) & quantization.
- Sparse extension: mask for interior zeros.
- Autograd / differentiation layer atop semirings.

---

**End of Document.** This specification defines the precise semantics and engineering constraints required for a high‑performance, correct VSLA C implementation. Deviations must be documented *prior* to merging into mainline.

