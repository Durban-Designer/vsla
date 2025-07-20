# TensorFlow Ragged Tensors

**Key Publication:** [The CoRa Tensor Compiler: Compilation for Ragged Tensors with Minimal Padding](https://proceedings.mlsys.org/paper/2021/file/138bb06965991a99af328644a53cc7de-Paper.pdf)

**Official Documentation:** [https://www.tensorflow.org/guide/ragged_tensor](https://www.tensorflow.org/guide/ragged_tensor)

**Blog Post:** [Introducing Ragged Tensors](https://blog.tensorflow.org/2018/11/introducing-ragged-tensors.html)

## Summary

Ragged tensors are the TensorFlow equivalent of nested variable-length lists. They are designed to handle data with non-uniform shapes efficiently. A ragged tensor is a tensor with one or more "ragged" dimensions, which are dimensions where slices can have different lengths. They are particularly useful in Natural Language Processing (NLP) for handling sentences of varying lengths and in time series analysis for data with missing entries. Ragged tensors support a wide range of TensorFlow operations, including mathematical, array, and string manipulation operations.
