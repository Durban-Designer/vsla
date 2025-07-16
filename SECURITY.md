# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in VSLA, please report it privately.

### How to Report

**Email**: royce.birnbaum@gmail.com
**Subject**: [SECURITY] VSLA Vulnerability Report

### What to Include

Please include the following information in your report:

1. **Description**: Clear description of the vulnerability
2. **Impact**: Potential impact and attack scenarios
3. **Reproduction**: Step-by-step instructions to reproduce
4. **Environment**: Operating system, compiler, and library versions
5. **Fix Suggestions**: Any potential fixes you might suggest (optional)

### Response Timeline

- **Acknowledgment**: Within 72 hours of your report
- **Initial Assessment**: Within 1 week 
- **Status Updates**: Weekly updates on progress
- **Resolution**: Target fix within 30 days for critical issues

### Security Scope

This security policy covers:

- **Memory safety**: Buffer overflows, use-after-free, memory leaks
- **Input validation**: Malformed tensor data, oversized inputs
- **Integer overflow**: Arithmetic operations, index calculations
- **File I/O**: Malicious tensor files, path traversal
- **Build system**: Dependency vulnerabilities, supply chain issues

### Out of Scope

The following are generally out of scope:

- Vulnerabilities in third-party dependencies (report to respective projects)
- Issues requiring physical access to the machine
- Social engineering attacks
- DoS attacks through excessive resource consumption (expected behavior)

### Disclosure Policy

- We follow responsible disclosure practices
- We will coordinate with you on disclosure timing
- We prefer 90 days from initial report to public disclosure
- Critical vulnerabilities may need faster disclosure

### Recognition

We maintain a security acknowledgments section for researchers who help improve VSLA's security. With your permission, we'll include:

- Your name or handle
- Brief description of the issue
- Date of the report

Thank you for helping keep VSLA secure!

## Security Best Practices

When using VSLA in your projects:

### Input Validation
```c
// Always validate tensor dimensions
if (tensor->rank > MAX_SUPPORTED_RANK) {
    return VSLA_ERROR_INVALID_ARGUMENT;
}

// Check for dimension overflow
uint64_t total_size = 1;
for (int i = 0; i < tensor->rank; i++) {
    if (tensor->shape[i] > MAX_DIMENSION_SIZE) {
        return VSLA_ERROR_INVALID_ARGUMENT;
    }
    // Check for multiplication overflow
    if (total_size > UINT64_MAX / tensor->shape[i]) {
        return VSLA_ERROR_OVERFLOW;
    }
    total_size *= tensor->shape[i];
}
```

### Memory Management
```c
// Always check allocation success
vsla_tensor_t* tensor = vsla_new(rank, shape, model, dtype);
if (!tensor) {
    // Handle allocation failure
    return NULL;
}

// Pair every allocation with deallocation
vsla_free(tensor);
```

### File I/O
```c
// Validate file size before loading
struct stat file_stat;
if (stat(filename, &file_stat) != 0) {
    return VSLA_ERROR_IO;
}
if (file_stat.st_size > MAX_SAFE_FILE_SIZE) {
    return VSLA_ERROR_FILE_TOO_LARGE;
}

// Use bounded reads
vsla_tensor_t* tensor = vsla_load(filename);
```

### Compilation Flags
```bash
# Enable security hardening
CFLAGS="-D_FORTIFY_SOURCE=2 -fstack-protector-strong -fPIE"
LDFLAGS="-Wl,-z,relro -Wl,-z,now -pie"

# Enable runtime checks in debug builds
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
```