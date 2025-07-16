# Autograd Module Memory Corruption Issue

## Summary
The `test_scaling_backward` test in the autograd module causes memory corruption when enabled, resulting in various malloc/free errors including:
- "corrupted size vs. prev_size in fastbins"
- "malloc(): unaligned fastbin chunk detected 2"
- Segmentation faults

## Current Status
- 7 out of 8 autograd tests pass successfully
- The `test_scaling_backward` test is currently disabled with a TODO comment
- All other modules (45 total tests) pass without issues

## How to Reproduce

### 1. Enable the Failing Test
Edit `/home/kenth56/vsla/tests/test_autograd.c` line 373:
```c
// Change from:
// RUN_TEST(test_scaling_backward);  // TODO: Fix memory corruption

// To:
RUN_TEST(test_scaling_backward);
```

### 2. Rebuild the Test
```bash
cd /home/kenth56/vsla/build/tests
gcc -std=c99 -Wall -Wextra -I../../include -c ../../tests/test_autograd.c -o CMakeFiles/vsla_tests.dir/test_autograd.c.o
gcc CMakeFiles/vsla_tests.dir/*.o ../libvsla.a -lm -o vsla_tests
```

### 3. Run the Test
```bash
./vsla_tests -s autograd
```

### 4. Expected Error
The test suite will crash during `test_scaling_backward` with one of these errors:
- "corrupted size vs. prev_size in fastbins"
- "malloc(): unaligned fastbin chunk detected 2"
- Segmentation fault (core dumped)

## Relevant Files

### 1. Test File
**File**: `/home/kenth56/vsla/tests/test_autograd.c`
**Function**: `test_scaling_backward` (lines 215-276)

This test:
- Creates two tensors a=[2,3] and b
- Computes b = 5 * a using vsla_scale
- Records the operation on the tape
- Sets output gradient to ones
- Calls backward pass
- Checks that input gradient is [5,5]

### 2. Autograd Implementation
**File**: `/home/kenth56/vsla/src/vsla_autograd.c`
**Key Functions**:
- `vsla_scale_backward` (lines 249-274) - Computes gradients for scaling operation
- `backward_operation` (lines 276-396) - Dispatches to specific backward functions
- `vsla_set_gradient` (lines 151-191) - Stores gradients with dynamic resizing
- `vsla_tape_free` (lines 53-72) - Cleans up tape memory

### 3. Header File
**File**: `/home/kenth56/vsla/include/vsla/vsla_autograd.h`
**Structure**: `vsla_tape_t` (lines 48-55) - Contains gradient storage array

## Technical Analysis

### Memory Management Design
The autograd module uses a dynamic array to store tensor-gradient pairs:
- Even indices (0, 2, 4...): Store tensor pointers (not owned)
- Odd indices (1, 3, 5...): Store gradient tensors (owned, must be freed)

### Suspected Issues
1. **Gradient Array Management**: The gradient array uses a paired storage system that may have alignment issues
2. **Memory Ownership**: Complex ownership model where some pointers are owned and others aren't
3. **Scale Backward Function**: Creates temporary tensors that may have lifetime issues
4. **Dynamic Resizing**: The realloc operation for gradient array may not preserve memory correctly

### Debugging Attempts Made
1. Added NULL checks after gradient retrieval
2. Fixed variable naming conflicts (grad_a vs zero_grad)
3. Ensured proper gradient initialization before use
4. Fixed scale operation to use correct source/destination

### Remaining Issues
The memory corruption appears to be related to:
- The gradient storage array management
- Possible double-free when cleaning up gradients
- Alignment issues with the paired storage approach

## Recommended Next Steps

1. **Use Valgrind** to get detailed memory error information:
   ```bash
   valgrind --leak-check=full --show-leak-kinds=all ./vsla_tests -s autograd
   ```

2. **Simplify Gradient Storage**: Consider using a hash table or separate arrays instead of the paired array approach

3. **Add Debug Logging**: Insert printf statements in:
   - `vsla_set_gradient` when storing gradients
   - `vsla_tape_free` when freeing gradients
   - `vsla_scale_backward` at each step

4. **Check Tensor Lifecycle**: Ensure tensors aren't being freed while still referenced in the gradient array

5. **Memory Sanitizer**: Compile with AddressSanitizer:
   ```bash
   gcc -fsanitize=address -g -std=c99 -Wall -Wextra -I../../include -c ../../tests/test_autograd.c
   ```

## Workaround
The test is currently disabled, allowing all other functionality to work correctly. The autograd module supports:
- Tape creation and management
- Operation recording
- Addition backward pass
- Subtraction backward pass
- Basic gradient management

Only the scaling backward pass has issues, likely due to the more complex memory management required.