# PyTorch Nested Tensors

**Official Documentation:** [https://pytorch.org/docs/stable/nested.html](https://pytorch.org/docs/stable/nested.html)

**Tutorial:** [https://pytorch.org/tutorials/prototype/nestedtensor.html](https://pytorch.org/tutorials/prototype/nestedtensor.html)

## Summary

Nested Tensors in PyTorch are designed to handle variable-length and irregularly shaped data, which is a common challenge in fields like Natural Language Processing (NLP) and Computer Vision (CV). They provide an efficient way to store and operate on "ragged" data, such as sentences of different lengths or images of varying sizes, within a single tensor-like structure. This avoids the need for inefficient padding and masking techniques. The feature is currently in a prototype stage, meaning the API is subject to change as it undergoes active development. A minimal version of NestedTensors has been integrated into the core PyTorch library.
