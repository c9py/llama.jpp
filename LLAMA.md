# llama.jpp - Pure J/J++ Implementation of llama.cpp

A pure J/J++ implementation of llama.cpp for running large language models locally.

## Overview

llama.jpp is a J programming language implementation of the core llama.cpp functionality, enabling efficient inference of large language models (LLMs) without external dependencies beyond the J interpreter. This implementation focuses on the essential components of transformer-based language models, including:

- **Quantization support**: Q4_0, Q8_0, F16, F32 formats
- **Transformer architecture**: Multi-head attention, feed-forward networks
- **Position embeddings**: RoPE (Rotary Position Embeddings)
- **Normalization**: RMS Layer Normalization
- **Efficient matrix operations**: Using J's native array operations
- **GGUF model format**: Support for loading quantized models

## Features

### Core Components

1. **Tensor Operations**
   - Matrix multiplication (`matmul`)
   - Transpose operations
   - Softmax activation
   - GELU activation function
   - Element-wise operations

2. **Quantization**
   - Q4_0: 4-bit quantization with 32-value blocks
   - Q8_0: 8-bit quantization
   - Dequantization to float32
   - Support for quantized model weights

3. **Transformer Layers**
   - Multi-head self-attention
   - Position-wise feed-forward networks
   - Residual connections
   - Layer normalization (RMS norm)

4. **Position Embeddings**
   - RoPE (Rotary Position Embeddings)
   - Configurable base frequency
   - Efficient frequency computation

5. **Model Loading**
   - GGUF format header parsing
   - Model structure initialization
   - Hyperparameter configuration

6. **Tokenization**
   - Simple BPE-style encoding
   - Vocabulary management
   - Token ID encoding/decoding

7. **Inference**
   - Autoregressive generation
   - Temperature-based sampling
   - Forward pass through transformer
   - Text generation

## Installation

### Requirements

- J Programming Language (version 9.0 or higher)
- For jpp features: jpp2.ijs (included in this repository)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/c9py/llama.jpp.git
cd llama.jpp
```

2. Load in J console:
```j
require '/path/to/llama.jpp/llama.ijs'
```

## Usage

### Basic Example

```j
NB. Load the library
require 'llama.ijs'

NB. Initialize a model
model =. llama_init 'path/to/model.gguf'

NB. Generate text
output =. model llama_infer 'Once upon a time'

NB. Generate with custom parameters
output =. model llama_generate ('Hello, world' ; 200 ; 0.7)
```

### Component Examples

#### Matrix Operations
```j
NB. Matrix multiplication
a =. 2 3 $ 1 2 3 4 5 6
b =. 3 2 $ 1 2 3 4 5 6
result =. a matmul_llama_ b
```

#### Softmax Activation
```j
x =. 1 2 3 4 5
probs =. softmax_llama_ x
NB. Output sums to 1.0
```

#### GELU Activation
```j
x =. _2 _1 0 1 2
activated =. gelu_llama_ x
```

#### RoPE (Rotary Position Embeddings)
```j
dim =. 128
freqs =. rope_freqs_llama_ dim ; 10000
```

#### Quantization
```j
NB. Q4_0 dequantization
scales =. 0.5 0.5 0.5 0.5
qvals =. 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
dequantized =. dequant_q4_0_llama_ scales ; qvals
```

#### RMS Normalization
```j
x =. 1 2 3 4 5
weight =. 1 1 1 1 1
eps =. 1e_5
normalized =. x rms_norm_llama_ (weight ; eps)
```

#### Attention Mechanism
```j
seq_len =. 4
head_dim =. 8
q =. ? seq_len head_dim $ 0
k =. ? seq_len head_dim $ 0
v =. ? seq_len head_dim $ 0
result =. (q ; k ; v ; a:) attention_llama_ (1 ; head_dim)
```

## Architecture

### File Structure

```
llama.jpp/
├── llama.ijs          # Main implementation
├── llama_test.ijs     # Test suite
├── llama_example.ijs  # Usage examples
├── LLAMA.md           # This documentation
├── jpp2.ijs           # J++ parser extensions
├── doubleadverb2.ijs  # Double adverb utilities
├── continuations.ijs  # Continuation support
└── fsm.ijs            # Finite state machine utilities
```

### Code Organization

The implementation is organized into logical sections:

1. **Constants and Configuration**: Model hyperparameters and type definitions
2. **Utility Functions**: Basic math and linear algebra operations
3. **Quantization**: Conversion between quantized and float formats
4. **Attention Mechanism**: Multi-head self-attention implementation
5. **Feed Forward Network**: Position-wise FFN with activations
6. **Layer Normalization**: RMS normalization
7. **Transformer Layer**: Complete transformer block
8. **Tokenization**: Text encoding and decoding
9. **Model Loading**: GGUF format parsing
10. **Inference Engine**: Generation and sampling
11. **Public API**: User-facing functions

## Implementation Details

### Quantization Formats

#### Q4_0
- 4-bit weights per value
- Block size: 32
- Each block contains: 1 float16 scale + 32 4-bit values
- Values range: -8 to 7

#### Q8_0
- 8-bit weights per value
- Block size: 32
- Each block contains: 1 float16 scale + 32 8-bit values
- Values range: -128 to 127

### Transformer Architecture

The implementation follows the standard transformer architecture:

1. **Input Embeddings**: Token embeddings + position embeddings
2. **Transformer Layers** (repeated N times):
   - RMS Layer Normalization
   - Multi-head Self-Attention
   - Residual Connection
   - RMS Layer Normalization
   - Feed-Forward Network
   - Residual Connection
3. **Output Layer**: Final normalization + output projection

### Attention Mechanism

Multi-head self-attention with:
- Query, Key, Value projections
- Scaled dot-product attention
- Softmax over attention scores
- Optional masking for causal attention

### RoPE (Rotary Position Embeddings)

Instead of absolute position embeddings, RoPE applies rotations to query and key vectors:
- Frequency computation based on dimension and base (typically 10000)
- Sine and cosine position encodings
- Applied before attention computation

## Performance

The J implementation leverages:
- Native J array operations for efficiency
- Rank polymorphism for broadcasting
- Tacit programming where appropriate
- Lazy evaluation for memory efficiency

Note: For production use with large models, consider:
- Using quantized models (Q4_0 or Q8_0)
- Reducing context length for memory constraints
- Batching operations when possible

## Testing

Run the test suite:

```bash
jconsole llama_test.ijs
```

Or run examples:

```bash
jconsole llama_example.ijs
```

Tests cover:
- Matrix operations
- Activation functions
- Quantization/dequantization
- RoPE embeddings
- Attention mechanisms
- Feed-forward networks
- Normalization
- Tokenization
- Sampling

## Limitations

This is a pure J implementation with the following limitations:

1. **Model Loading**: GGUF parsing is simplified; full binary format support requires additional work
2. **Performance**: Pure J is slower than optimized C++; suitable for small models and educational purposes
3. **Memory**: Large models may exceed available memory without careful optimization
4. **Hardware Acceleration**: No GPU support in this implementation
5. **Features**: Focus on core inference; training and fine-tuning not supported

## Future Enhancements

Potential improvements:

- [ ] Complete GGUF binary format parsing
- [ ] Support for more quantization formats (Q5, Q6)
- [ ] KV cache for faster generation
- [ ] Grouped-query attention (GQA)
- [ ] Flash attention optimization
- [ ] Model-specific configurations
- [ ] More tokenization strategies
- [ ] Streaming generation
- [ ] Batch inference
- [ ] Performance profiling and optimization

## Contributing

Contributions are welcome! Areas for contribution:

- Performance optimizations
- Additional quantization formats
- Better tokenization
- More comprehensive testing
- Documentation improvements
- Example models and use cases

## License

GPL3 - Same as jpp base library

## References

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Original C++ implementation
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) - Model format specification
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [RoFormer](https://arxiv.org/abs/2104.09864) - RoPE position embeddings
- [J Programming Language](https://www.jsoftware.com/) - J language reference

## Examples

See `llama_example.ijs` for comprehensive examples of all components.

## Support

For issues and questions:
- GitHub Issues: [https://github.com/c9py/llama.jpp/issues](https://github.com/c9py/llama.jpp/issues)
- J Programming Forum: [https://www.jsoftware.com/forums/](https://www.jsoftware.com/forums/)

## Acknowledgments

- Georgi Gerganov for llama.cpp
- J Software for the J programming language
- The jpp (J++) community for parser extensions
