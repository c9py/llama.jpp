NB. llama_example.ijs - Example usage of llama.jpp
NB. Demonstrates the basic API and capabilities

NB. Load the llama.jpp library
require '/home/runner/work/llama.jpp/llama.jpp/llama.ijs'

NB. =========================================================================
NB. Example 1: Basic Math Operations
NB. =========================================================================

smoutput 'Example 1: Matrix Multiplication'
smoutput '--------------------------------'
a =. 2 3 $ 1 2 3 4 5 6
smoutput 'Matrix A:'
smoutput a
b =. 3 2 $ 1 2 3 4 5 6
smoutput 'Matrix B:'
smoutput b
result =. a matmul_llama_ b
smoutput 'A x B:'
smoutput result
smoutput ''

NB. =========================================================================
NB. Example 2: Softmax
NB. =========================================================================

smoutput 'Example 2: Softmax Activation'
smoutput '------------------------------'
x =. 1 2 3 4 5
smoutput 'Input:'
smoutput x
probs =. softmax_llama_ x
smoutput 'Softmax output (probabilities):'
smoutput probs
smoutput 'Sum of probabilities:'
smoutput +/ probs
smoutput ''

NB. =========================================================================
NB. Example 3: GELU Activation
NB. =========================================================================

smoutput 'Example 3: GELU Activation'
smoutput '---------------------------'
x =. _2 _1 0 1 2
smoutput 'Input:'
smoutput x
activated =. gelu_llama_ x
smoutput 'GELU output:'
smoutput activated
smoutput ''

NB. =========================================================================
NB. Example 4: RoPE Frequencies
NB. =========================================================================

smoutput 'Example 4: RoPE (Rotary Position Embeddings)'
smoutput '--------------------------------------------'
dim =. 64
smoutput 'Dimension: ', ": dim
freqs =. rope_freqs_llama_ dim ; 10000
smoutput 'Number of frequencies: ', ": # freqs
smoutput 'First few frequencies:'
smoutput 5 {. freqs
smoutput ''

NB. =========================================================================
NB. Example 5: Quantization
NB. =========================================================================

smoutput 'Example 5: Quantization and Dequantization'
smoutput '-------------------------------------------'
scales =. 0.5 0.5 0.5 0.5
qvals =. 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
smoutput 'Quantized values:'
smoutput qvals
dequantized =. dequant_q4_0_llama_ scales ; qvals
smoutput 'Dequantized values (Q4_0):'
smoutput dequantized
smoutput ''

NB. =========================================================================
NB. Example 6: RMS Normalization
NB. =========================================================================

smoutput 'Example 6: RMS Layer Normalization'
smoutput '-----------------------------------'
x =. 1 2 3 4 5
weight =. 1 1 1 1 1
eps =. 1e_5
smoutput 'Input:'
smoutput x
normalized =. x rms_norm_llama_ (weight ; eps)
smoutput 'Normalized output:'
smoutput normalized
smoutput ''

NB. =========================================================================
NB. Example 7: Model Loading (Structure)
NB. =========================================================================

smoutput 'Example 7: Model Structure'
smoutput '---------------------------'
model =. load_model_llama_ 'example.gguf'
smoutput 'Model structure created (placeholder)'
smoutput 'Model parameters:'
smoutput model
smoutput ''

NB. =========================================================================
NB. Example 8: Tokenization
NB. =========================================================================

smoutput 'Example 8: Simple Tokenization'
smoutput '-------------------------------'
vocab =. ;: 'the quick brown fox jumps over the lazy dog'
smoutput 'Vocabulary:'
smoutput vocab
text =. 'quick brown fox'
smoutput 'Text to encode: ', text
ids =. vocab encode_llama_ text
smoutput 'Token IDs:'
smoutput ids
decoded =. vocab decode_llama_ ids
smoutput 'Decoded text:', decoded
smoutput ''

NB. =========================================================================
NB. Example 9: Token Sampling
NB. =========================================================================

smoutput 'Example 9: Token Sampling'
smoutput '--------------------------'
logits =. 1 2 3 4 5
temp =. 1.0
smoutput 'Logits:'
smoutput logits
token =. logits sample_token_llama_ temp
smoutput 'Sampled token index: ', ": token
smoutput ''

NB. =========================================================================
NB. Summary
NB. =========================================================================

smoutput '=========================================='
smoutput 'llama.jpp Examples Complete'
smoutput '=========================================='
smoutput ''
smoutput 'This demonstrates the core components of llama.jpp:'
smoutput '  - Matrix operations for neural network computations'
smoutput '  - Activation functions (GELU, Softmax)'
smoutput '  - Quantization support (Q4_0, Q8_0)'
smoutput '  - RoPE (Rotary Position Embeddings)'
smoutput '  - Layer normalization (RMS norm)'
smoutput '  - Model structure and loading'
smoutput '  - Tokenization'
smoutput '  - Sampling'
smoutput ''
smoutput 'For full LLM inference, use:'
smoutput '  model =. llama_init ''path/to/model.gguf'''
smoutput '  output =. model llama_infer ''Your prompt here'''
smoutput ''
