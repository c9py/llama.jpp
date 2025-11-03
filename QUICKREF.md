# llama.jpp Quick Reference

## Installation

```bash
git clone https://github.com/c9py/llama.jpp.git
cd llama.jpp
```

## Basic Usage

### Load the library
```j
require 'llama.ijs'
```

### Initialize a model
```j
model =. llama_init 'path/to/model.gguf'
```

### Generate text
```j
NB. Simple inference
output =. model llama_infer 'Once upon a time'

NB. With custom parameters (prompt ; max_tokens ; temperature)
output =. model llama_generate ('Hello, world' ; 200 ; 0.7)
```

## Core Components

### Matrix Operations
```j
NB. Matrix multiplication
result =. matrix1 matmul_llama_ matrix2

NB. Transpose
transposed =. transpose_llama_ matrix

NB. Normalize
normalized =. normalize_llama_ vector
```

### Activation Functions
```j
NB. Softmax (produces probability distribution)
probs =. softmax_llama_ logits

NB. GELU activation
activated =. gelu_llama_ input
```

### Quantization
```j
NB. Dequantize Q4_0 format
values =. GGML_TYPE_Q4_0_llama_ dequantize_llama_ (scales ; quantized_data)

NB. Dequantize Q8_0 format
values =. GGML_TYPE_Q8_0_llama_ dequantize_llama_ (scales ; quantized_data)
```

### Position Embeddings (RoPE)
```j
NB. Compute RoPE frequencies
freqs =. rope_freqs_llama_ dimension ; base_frequency

NB. Apply RoPE to queries
rotated =. queries apply_rope_llama_ (positions ; dimension)
```

### Normalization
```j
NB. RMS Layer Normalization
normalized =. input rms_norm_llama_ (weights ; epsilon)
```

### Attention
```j
NB. Multi-head attention
NB. Input: (query ; key ; value ; mask)
NB. Config: (n_heads ; head_dim)
output =. (q ; k ; v ; mask) attention_llama_ (n_heads ; head_dim)
```

### Feed-Forward Network
```j
NB. Two-layer FFN with GELU
NB. Params: (w1 ; w2 ; b1 ; b2)
output =. input ffn_llama_ params
```

### Transformer Layer
```j
NB. Complete transformer block
NB. Params: (attention_weights ; ffn_weights ; norm1 ; norm2)
output =. input transformer_layer_llama_ layer_params
```

### Tokenization
```j
NB. Encode text to token IDs
ids =. vocab encode_llama_ text

NB. Decode token IDs to text
text =. vocab decode_llama_ ids
```

### Sampling
```j
NB. Sample next token with temperature
token_id =. logits sample_token_llama_ temperature
```

## Model Configuration

Default hyperparameters:
```j
DEFAULT_N_VOCAB_llama_  =: 32000   NB. Vocabulary size
DEFAULT_N_CTX_llama_    =: 2048    NB. Context length
DEFAULT_N_EMBD_llama_   =: 4096    NB. Embedding dimension
DEFAULT_N_HEAD_llama_   =: 32      NB. Number of attention heads
DEFAULT_N_LAYER_llama_  =: 32      NB. Number of transformer layers
```

## Quantization Types

```j
GGML_TYPE_F32_llama_  =: 0   NB. 32-bit float
GGML_TYPE_F16_llama_  =: 1   NB. 16-bit float
GGML_TYPE_Q4_0_llama_ =: 2   NB. 4-bit quantization
GGML_TYPE_Q4_1_llama_ =: 3   NB. 4-bit quantization (variant)
GGML_TYPE_Q8_0_llama_ =: 8   NB. 8-bit quantization
```

## Examples

### Example 1: Simple Math
```j
NB. Matrix multiplication
a =. 2 3 $ 1 2 3 4 5 6
b =. 3 2 $ 1 2 3 4 5 6
result =. a matmul_llama_ b
NB. result is: 22 28
NB.             49 64
```

### Example 2: Softmax
```j
x =. 1 2 3 4 5
probs =. softmax_llama_ x
NB. probs sums to 1.0
+/ probs  NB. 1
```

### Example 3: RoPE Embeddings
```j
dim =. 128
base =. 10000
freqs =. rope_freqs_llama_ dim ; base
NB. Returns 64 frequencies (half of dim)
```

### Example 4: Attention
```j
NB. Create sample query, key, value matrices
seq_len =. 8
head_dim =. 64
q =. ? seq_len head_dim $ 0
k =. ? seq_len head_dim $ 0
v =. ? seq_len head_dim $ 0

NB. Apply attention
output =. (q ; k ; v ; a:) attention_llama_ (1 ; head_dim)
```

### Example 5: Complete Inference Pipeline
```j
NB. Load model
model =. llama_init 'llama-7b-q4_0.gguf'

NB. Prepare prompt
prompt =. 'Explain quantum computing in simple terms:'

NB. Generate response (max 150 tokens, temperature 0.8)
response =. model llama_generate (prompt ; 150 ; 0.8)

NB. Display result
smoutput response
```

## Testing

Run the test suite:
```bash
jconsole llama_test.ijs
```

Run examples:
```bash
jconsole llama_example.ijs
```

Validate installation:
```bash
./validate.sh
```

## Performance Tips

1. **Use quantized models**: Q4_0 and Q8_0 formats are much faster
2. **Reduce context length**: Smaller `n_ctx` uses less memory
3. **Lower temperature**: Values closer to 0.0 are more deterministic and faster
4. **Batch operations**: Process multiple sequences together when possible

## Architecture Overview

```
Input Text
    ↓
[Tokenization]
    ↓
Token IDs
    ↓
[Embedding Layer]
    ↓
[Transformer Layers] x N
  ├─ RMS Norm
  ├─ Multi-Head Attention
  │   ├─ Query/Key/Value Projections
  │   ├─ RoPE Position Embeddings
  │   ├─ Scaled Dot-Product Attention
  │   └─ Output Projection
  ├─ Residual Connection
  ├─ RMS Norm
  ├─ Feed-Forward Network
  │   ├─ Linear (expand)
  │   ├─ GELU Activation
  │   └─ Linear (contract)
  └─ Residual Connection
    ↓
[Final Layer Norm]
    ↓
[Output Projection]
    ↓
Logits
    ↓
[Sampling]
    ↓
Output Text
```

## Comparison with llama.cpp

| Feature | llama.cpp | llama.jpp |
|---------|-----------|-----------|
| Language | C++ | J |
| Dependencies | None | J runtime |
| GPU Support | Yes | No |
| Quantization | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, etc. | Q4_0, Q8_0 |
| Model Format | GGUF | GGUF (partial) |
| Performance | Very Fast | Moderate |
| Use Case | Production | Education/Research |
| Code Size | ~50K LOC | ~800 LOC |

## Limitations

- **Performance**: Slower than C++ implementation
- **Model Size**: Limited by available memory
- **Features**: Core functionality only
- **GPU**: No GPU acceleration
- **Models**: Works best with small models (<7B parameters)

## Contributing

Contributions welcome! Key areas:

- Performance optimization
- Additional quantization formats
- Complete GGUF parser
- More tokenization strategies
- GPU support (via J addons)
- Example models and notebooks

## License

GPL3 (same as jpp base)

## Resources

- [Full Documentation](LLAMA.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [J Programming Language](https://www.jsoftware.com/)
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

## Support

- GitHub Issues: https://github.com/c9py/llama.jpp/issues
- J Forum: https://www.jsoftware.com/forums/
