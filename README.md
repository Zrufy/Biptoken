# Biptoken

A fast BPE tokenizer with perfect text reconstruction guarantee.

## Features

- Perfect reconstruction: Always decodes to the exact original text, preserving all whitespace and formatting
- Fast: 2.24x faster encoding and 1.74x faster decoding than tiktoken on typical workloads
- Special token support: Built-in handling for common special tokens (`<user>`, <assistant>, etc.)
- Simple API: Compatible with common tokenizer interfaces

## Installation

```bash
# Clone the repository
git clone https://github.com/Zrufy/biptoken.git
cd biptoken

# Install in development mode
pip install -e .

# Or install directly
pip install .
```

## Quick Start

```python
from biptoken import Biptoken

# Create and train a tokenizer
tokenizer = Biptoken(vocab_size=32000)
tokenizer.train(["Your training texts here"])

# Encode and decode
text = "Hello world!   Multiple spaces preserved."
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)
assert text == decoded  # Always true
```

## Training from File

```python
tokenizer = Biptoken(vocab_size=50000)
tokenizer.train_from_file("path/to/your/text.txt")
tokenizer.save("my_tokenizer.json")
```

## Performance

Benchmark on 25M character text (5 iterations):

|Metric                |Biptoken|tiktoken|Comparison                                      |
|----------------------|------------|--------|------------------------------------------------|
|Encoding speed        |0.85s       |2.31s   |2.7x faster                                    |
|Decoding speed        |0.08s       |0.20s   |2.5x faster                                    |
|Token count           |9.6M       |7.2M    |~1.3x more tokens                                  |
|Perfect reconstruction|✓           |✓*      |Both achieve perfect reconstruction on this test|

*Note: tiktoken does not guarantee perfect reconstruction for all edge cases (multiple spaces, special formatting).

## Design Trade-offs

Biptoken optimizes for:

- Speed: Faster encoding/decoding through optimized data structures
- Correctness: Guaranteed perfect text reconstruction
- Simplicity: Straightforward implementation without complex normalizations

This comes at the cost of token efficiency - Biptoken typically produces ~2x more tokens than tiktoken.

## Use Cases

Biptoken is ideal for applications where:

- Perfect text reconstruction is critical (code editors, document processing)
- Tokenization speed matters more than token count
- Whitespace and formatting must be preserved exactly

For LLM applications with strict token limits, consider using tiktoken or similar tokenizers optimized for token efficiency.

## API Reference

### Biptoken(vocab_size=32000)

Create a new tokenizer with specified vocabulary size.

### train(texts, min_freq=2)

Train the tokenizer on a list of texts.

### train_from_file(filepath, min_freq=2)

Train the tokenizer from a text file.

### encode(text, add_special_tokens=True)

Encode text to token IDs.

### decode(ids, skip_special_tokens=True)

Decode token IDs back to text.

### save(filepath)

Save tokenizer to JSON file.

### load(filepath)

Load tokenizer from JSON file.

## Implementation Details

Biptoken uses:

- Byte Pair Encoding (BPE) for subword tokenization
- Special tokens for preserving spaces (`<space>`) and uppercase markers (`<upper>`)
- LRU cache for common encoding/decoding operations
- Optimized data structures (defaultdict, pre-compiled regex patterns)

## Requirements

- Python 3.6+
- numpy
- regex

## License

MIT
