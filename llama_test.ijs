NB. llama_test.ijs - Test suite for llama.jpp
NB. Tests core functionality of the LLM inference implementation

require '/home/runner/work/llama.jpp/llama.jpp/llama.ijs'

NB. =========================================================================
NB. Test Utilities
NB. =========================================================================

test_assert =: 4 : 0
  name =. x
  result =. y
  if. result do.
    smoutput 'PASS: ', name
  else.
    smoutput 'FAIL: ', name
  end.
  result
)

NB. =========================================================================
NB. Test Basic Math Operations
NB. =========================================================================

test_matmul =: 3 : 0
  a =. 2 3 $ 1 2 3 4 5 6
  b =. 3 2 $ 1 2 3 4 5 6
  result =. a matmul_llama_ b
  expected =. 2 2 $ 22 28 49 64
  'Matrix multiplication' test_assert result -: expected
)

test_softmax =: 3 : 0
  x =. 1 2 3 4
  result =. softmax_llama_ x
  NB. Check that sum equals 1
  sum_check =. 1 = +/ result
  pos_check =. *./ result > 0
  'Softmax properties' test_assert sum_check *. pos_check
)

test_gelu =: 3 : 0
  x =. _2 _1 0 1 2
  result =. gelu_llama_ x
  NB. Check GELU at 0 is 0
  zero_check =. (|@({&result) 2) < 0.01
  'GELU activation' test_assert zero_check
)

NB. =========================================================================
NB. Test Quantization
NB. =========================================================================

test_dequant_q4_0 =: 3 : 0
  NB. Test Q4_0 dequantization
  scales =. 1 1 1 1
  qvals =. 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
  result =. dequant_q4_0_llama_ scales ; qvals
  NB. Check that values are in expected range
  'Q4_0 dequantization' test_assert (#result) = #qvals
)

test_dequant_q8_0 =: 3 : 0
  NB. Test Q8_0 dequantization
  scales =. 1 1
  qvals =. 0 64 128 192 255
  result =. dequant_q8_0_llama_ scales ; qvals
  NB. Check that dequantization produces values
  'Q8_0 dequantization' test_assert (#result) = #qvals
)

NB. =========================================================================
NB. Test RoPE (Rotary Position Embeddings)
NB. =========================================================================

test_rope_freqs =: 3 : 0
  dim =. 128
  freqs =. rope_freqs_llama_ dim ; 10000
  NB. Check that we get half-dimension frequencies
  'RoPE frequencies' test_assert (#freqs) = <. dim % 2
)

NB. =========================================================================
NB. Test Normalization
NB. =========================================================================

test_rms_norm =: 3 : 0
  x =. 1 2 3 4 5
  weight =. 1 1 1 1 1
  eps =. 1e_5
  result =. x rms_norm_llama_ (weight ; eps)
  NB. Check that output has same shape
  'RMS normalization' test_assert (#result) = #x
)

NB. =========================================================================
NB. Test Attention
NB. =========================================================================

test_attention =: 3 : 0
  NB. Simple attention test
  seq_len =. 4
  head_dim =. 8
  
  q =. ? seq_len head_dim $ 0
  k =. ? seq_len head_dim $ 0
  v =. ? seq_len head_dim $ 0
  mask =. a:
  
  result =. (q ; k ; v ; mask) attention_llama_ (1 ; head_dim)
  
  NB. Check output shape
  'Attention mechanism' test_assert ($ result) -: ($ q)
)

NB. =========================================================================
NB. Test Feed Forward Network
NB. =========================================================================

test_ffn =: 3 : 0
  input_dim =. 8
  hidden_dim =. 16
  
  x =. ? input_dim $ 0
  w1 =. ? input_dim hidden_dim $ 0
  w2 =. ? hidden_dim input_dim $ 0
  b1 =. ? hidden_dim $ 0
  b2 =. ? input_dim $ 0
  
  result =. x ffn_llama_ (w1 ; w2 ; b1 ; b2)
  
  NB. Check output shape
  'Feed forward network' test_assert (#result) = input_dim
)

NB. =========================================================================
NB. Test Model Loading
NB. =========================================================================

test_model_structure =: 3 : 0
  NB. Test model structure creation
  model =. load_model_llama_ 'nonexistent.gguf'
  NB. Should create a model structure even if file doesn't exist
  'Model structure' test_assert (L. model) > 0
)

NB. =========================================================================
NB. Test Tokenization
NB. =========================================================================

test_encode_decode =: 3 : 0
  vocab =. ;: 'the quick brown fox jumps over lazy dog'
  text =. 'quick brown fox'
  
  NB. Encode
  ids =. vocab encode_llama_ text
  
  NB. Decode
  decoded =. vocab decode_llama_ ids
  
  NB. Check that encoding produces IDs
  'Tokenization' test_assert (#ids) > 0
)

NB. =========================================================================
NB. Test Sampling
NB. =========================================================================

test_sampling =: 3 : 0
  logits =. 1 2 3 4 5
  temp =. 1.0
  
  token =. logits sample_token_llama_ temp
  
  NB. Check that sampled token is valid index
  'Token sampling' test_assert (token >: 0) *. (token < #logits)
)

NB. =========================================================================
NB. Integration Tests
NB. =========================================================================

test_forward_pass =: 3 : 0
  NB. Test a simple forward pass
  model =. load_model_llama_ 'test.gguf'
  input_ids =. 1 2 3
  pos =. 0
  
  result =. model forward_llama_ (input_ids ; pos)
  
  NB. Check that forward pass produces output
  'Forward pass' test_assert (#result) > 0
)

NB. =========================================================================
NB. Run All Tests
NB. =========================================================================

run_all_tests =: 3 : 0
  smoutput ''
  smoutput '=========================================='
  smoutput 'Running llama.jpp Test Suite'
  smoutput '=========================================='
  smoutput ''
  
  NB. Math operations
  smoutput 'Testing Math Operations...'
  test_matmul ''
  test_softmax ''
  test_gelu ''
  smoutput ''
  
  NB. Quantization
  smoutput 'Testing Quantization...'
  test_dequant_q4_0 ''
  test_dequant_q8_0 ''
  smoutput ''
  
  NB. RoPE
  smoutput 'Testing RoPE...'
  test_rope_freqs ''
  smoutput ''
  
  NB. Normalization
  smoutput 'Testing Normalization...'
  test_rms_norm ''
  smoutput ''
  
  NB. Attention
  smoutput 'Testing Attention...'
  test_attention ''
  smoutput ''
  
  NB. FFN
  smoutput 'Testing Feed Forward...'
  test_ffn ''
  smoutput ''
  
  NB. Model
  smoutput 'Testing Model Loading...'
  test_model_structure ''
  smoutput ''
  
  NB. Tokenization
  smoutput 'Testing Tokenization...'
  test_encode_decode ''
  smoutput ''
  
  NB. Sampling
  smoutput 'Testing Sampling...'
  test_sampling ''
  smoutput ''
  
  NB. Integration
  smoutput 'Testing Integration...'
  test_forward_pass ''
  smoutput ''
  
  smoutput '=========================================='
  smoutput 'Test Suite Complete'
  smoutput '=========================================='
)

NB. Auto-run tests when loaded
run_all_tests ''
