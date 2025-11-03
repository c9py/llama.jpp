# Installation Instructions for llama.jpp

## Prerequisites

1. **J Programming Language** (version 9.0 or higher)
   - Download from: https://www.jsoftware.com/
   - Or install via package manager:
     - Ubuntu/Debian: `sudo apt-get install jlang`
     - macOS: `brew install j`

2. **jpp (J Plus Plus)** - Already included in this repository
   - Files: `jpp2.ijs`, `doubleadverb2.ijs`, `continuations.ijs`, `fsm.ijs`

## Installation Methods

### Method 1: Direct Use (Recommended for testing)

1. Clone the repository:
```bash
git clone https://github.com/c9py/llama.jpp.git
cd llama.jpp
```

2. Load in J console:
```j
NB. From the repository directory
load 'llama.ijs'

NB. Test the installation
model =. llama_init 'example.gguf'
```

### Method 2: Install as J Addon

1. Clone to J's addon directory:
```bash
# Find your J installation directory
# Typical locations:
# Linux: ~/.local/share/j/addons/
# macOS: ~/Library/j/addons/
# Windows: C:\Users\YourName\Documents\J\addons\

cd ~/.local/share/j/addons/
git clone https://github.com/c9py/llama.jpp.git
```

2. Load from anywhere in J:
```j
require '~addons/llama.jpp/llama.ijs'
```

### Method 3: System-wide Installation

1. Copy to J's system addon directory:
```bash
# Linux/macOS (may require sudo)
sudo cp -r llama.jpp /usr/share/j/addons/

# Or symlink
sudo ln -s /path/to/llama.jpp /usr/share/j/addons/llama.jpp
```

2. Load from anywhere:
```j
require 'llama.jpp/llama.ijs'
```

## Verification

Run the validation script:
```bash
cd llama.jpp
chmod +x validate.sh
./validate.sh
```

Expected output:
```
✓ All required files present
✓ All key components implemented
✓ Documentation complete
```

## Quick Test

Create a test file `test.ijs`:
```j
NB. Load library
load 'llama.ijs'

NB. Test basic operations
smoutput 'Testing matrix multiply...'
a =. 2 3 $ 1 2 3 4 5 6
b =. 3 2 $ 1 2 3 4 5 6
result =. a matmul_llama_ b
smoutput result

smoutput 'Testing softmax...'
x =. 1 2 3 4 5
probs =. softmax_llama_ x
smoutput probs
smoutput 'Sum of probabilities: ', ": +/ probs

smoutput 'llama.jpp working correctly!'
exit 0
```

Run it:
```bash
jconsole < test.ijs
```

## Usage in Your Scripts

### Option 1: Relative Path
If your script is in the same directory as llama.ijs:
```j
load 'llama.ijs'
```

### Option 2: Absolute Path
```j
load '/path/to/llama.jpp/llama.ijs'
```

### Option 3: Addon Path
If installed as addon:
```j
require '~addons/llama.jpp/llama.ijs'
```

## Troubleshooting

### "File not found" Error
- Check that llama.ijs is in the correct location
- Use absolute path if relative path doesn't work
- Verify J can access the file: `1!:4 <'llama.ijs'`

### "Name not defined" Errors
- Some functions may require jpp to be loaded first
- Load jpp2.ijs if using J++ syntax features
- Check that you're using the correct locale (functions are in 'llama' locale)

### Interactive Mode Hangs
- This is normal behavior for jconsole when reading from stdin
- Use `exit 0` at the end of scripts
- Or run examples directly: `jconsole llama_example.ijs`

### Missing Dependencies
If you get errors about undefined functions:
```j
NB. Load required J libraries
require 'files'      NB. For file operations
require 'numeric'    NB. For numeric operations
```

## Running Examples

### Example Scripts
```bash
# Run comprehensive examples
jconsole llama_example.ijs

# Run test suite
jconsole llama_test.ijs
```

### Interactive Use
```bash
jconsole
```

Then in the J console:
```j
   load 'llama.ijs'
   
   NB. Test a simple operation
   2 3 $ 1 2 3 4 5 6 matmul_llama_ 3 2 $ 1 2 3 4 5 6
22 28
49 64
   
   NB. Exit
   exit 0
```

## Model Files

To use actual LLaMA models:

1. Download a GGUF model file (e.g., from Hugging Face)
2. Place it in an accessible location
3. Load and use:
```j
model =. llama_init '/path/to/model.gguf'
output =. model llama_infer 'Your prompt here'
```

Note: Current implementation has basic GGUF support. Full binary parsing is in development.

## Development Setup

For contributing or development:

1. Clone with all branches:
```bash
git clone https://github.com/c9py/llama.jpp.git
cd llama.jpp
```

2. Create a development script:
```j
NB. dev.ijs - Development helper
load 'llama.ijs'

NB. Reload on changes
reload =: 3 : 0
  smoutput 'Reloading llama.ijs...'
  load 'llama.ijs'
  smoutput 'Reloaded.'
)

NB. Quick test
test =: 3 : 0
  smoutput 'Running quick test...'
  a =. 2 3 $ i. 6
  b =. 3 2 $ i. 6
  smoutput a matmul_llama_ b
)
```

## Getting Models

Small quantized models for testing:
- TinyLlama (1.1B): ~600MB in Q4_0
- LLaMA 2 7B: ~3.5GB in Q4_0
- Download from Hugging Face: https://huggingface.co/models

## Performance Notes

- Pure J implementation is slower than C++
- Best for models under 7B parameters
- Use quantized models (Q4_0) for better performance
- Reduce context length if memory is limited

## Support

- Documentation: See LLAMA.md and QUICKREF.md
- Issues: https://github.com/c9py/llama.jpp/issues
- J Forum: https://www.jsoftware.com/forums/

## Next Steps

After installation:
1. Read QUICKREF.md for API reference
2. Run llama_example.ijs to see usage examples
3. Check LLAMA.md for detailed documentation
4. Try with small GGUF models
