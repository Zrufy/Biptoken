import time
import tiktoken
from biptoken import BipTokenizer
import numpy as np
import os

def benchmark_tokenizers(text, iterations=10):
    print(f"Benchmarking tokenizers on text of length {len(text)} characters")
    print(f"Running {iterations} iterations for each tokenizer")
    print("-" * 60)
    
    # Prepare the tokenizers
    bip_tokenizer = BipTokenizer(vocab_size=50000)
    # Train BipToken on a small sample
    sample_size = min(10000, len(text))
    print(f"Training BipTokenizer on a sample of {sample_size} characters...")
    bip_tokenizer.train([text[:sample_size]])
    
    # Load tiktoken with the cl100k_base model (usato per GPT-4)
    print("Loading tiktoken cl100k_base model...")
    tik_tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Benchmark di encoding
    print("\nENCODING BENCHMARK:")
    
    # BipToken encoding
    print("Running BipToken encoding...")
    start_time = time.time()
    for _ in range(iterations):
        bip_tokens = bip_tokenizer.encode(text)
    bip_encode_time = time.time() - start_time
    print(f"BipToken encoding: {bip_encode_time:.4f} seconds total, {bip_encode_time/iterations:.6f} seconds per iteration")
    
    # tiktoken encoding
    print("Running tiktoken encoding...")
    start_time = time.time()
    for _ in range(iterations):
        tik_tokens = tik_tokenizer.encode(text)
    tik_encode_time = time.time() - start_time
    print(f"tiktoken encoding: {tik_encode_time:.4f} seconds total, {tik_encode_time/iterations:.6f} seconds per iteration")
    
    if tik_encode_time > 0:
        if bip_encode_time > tik_encode_time:
            print(f"tiktoken is {bip_encode_time/tik_encode_time:.2f}x faster at encoding")
        else:
            print(f"BipToken is {tik_encode_time/bip_encode_time:.2f}x faster at encoding")
    else:
        print("tiktoken encoding time too small to measure accurately")
    
    # Benchmark di decoding
    print("\nDECODING BENCHMARK:")
    
    # BipToken decoding
    print("Running BipToken decoding...")
    start_time = time.time()
    for _ in range(iterations):
        bip_decoded = bip_tokenizer.decode(bip_tokens)
    bip_decode_time = time.time() - start_time
    print(f"BipToken decoding: {bip_decode_time:.4f} seconds total, {bip_decode_time/iterations:.6f} seconds per iteration")
    
    # tiktoken decoding
    print("Running tiktoken decoding...")
    start_time = time.time()
    for _ in range(iterations):
        tik_decoded = tik_tokenizer.decode(tik_tokens)
    tik_decode_time = time.time() - start_time
    print(f"tiktoken decoding: {tik_decode_time:.4f} seconds total, {tik_decode_time/iterations:.6f} seconds per iteration")
    
    if tik_decode_time > 0:
        if bip_decode_time > tik_decode_time:
            print(f"tiktoken is {bip_decode_time/tik_decode_time:.2f}x faster at decoding")
        else:
            print(f"BipToken is {tik_decode_time/bip_decode_time:.2f}x faster at decoding")
    else:
        print("tiktoken decoding time too small to measure accurately")
    
    # Perfect reconstruction check
    print("\nPERFECT RECONSTRUCTION TEST:")
    print(f"BipToken perfect reconstruction: {text == bip_decoded}")
    print(f"tiktoken perfect reconstruction: {text == tik_decoded}")
    
    # Token statistics
    print("\nTOKENIZATION STATISTICS:")
    print(f"BipToken tokens: {len(bip_tokens)}")
    print(f"tiktoken tokens: {len(tik_tokens)}")
    
    # Show examples of differences if reconstruction is not perfect
    if text != tik_decoded:
        print("\nExample of tiktoken reconstruction differences:")
        # Find the first difference
        for i, (orig_char, decoded_char) in enumerate(zip(text, tik_decoded)):
            if orig_char != decoded_char:
                start = max(0, i - 20)
                end = min(len(text), i + 20)
                print(f"Position {i}:")
                print(f"Original:  '...{text[start:end]}...'")
                print(f"Decoded:   '...{tik_decoded[start:end]}...'")
                print(f"Difference: Original '{orig_char}' vs Decoded '{decoded_char}'")
                break
        
        # If the strings have different lengths
        if len(text) != len(tik_decoded):
            print(f"Length difference: Original {len(text)} vs Decoded {len(tik_decoded)}")
    
    return {
        "bip_encoding_time": bip_encode_time/iterations,
        "tik_encoding_time": tik_encode_time/iterations,
        "bip_decoding_time": bip_decode_time/iterations,
        "tik_decoding_time": tik_decode_time/iterations,
        "bip_perfect": text == bip_decoded,
        "tik_perfect": text == tik_decoded,
        "bip_tokens": len(bip_tokens),
        "tik_tokens": len(tik_tokens)
    }

if __name__ == "__main__":
    # Path to the text file
    file_path = "your_path_to_file.txt"  # Change to the correct path
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Use a correct absolute or relative path.")
        file_path = input("Enter the path to the text file: ")
    
    # Read the text file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"File loaded successfully: {len(text)} characters")
    except Exception as e:
        print(f"Error reading the file: {e}")
        print("Generating random text as fallback...")
        np.random.seed(42)
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
                "hello", "world", "python", "is", "great", "for", "natural", 
                "language", "processing", "and", "machine", "learning", ".", "!"]
        text = " ".join(np.random.choice(words, size=10000))
    
    # Run the benchmark
    iterations = 5  # Reduced for large files
    results = benchmark_tokenizers(text, iterations=iterations)
    
    # Save the results to a file
    with open("benchmark_results.txt", "w", encoding="utf-8") as f:
        f.write("BENCHMARK RESULTS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Text length: {len(text)} characters\n")
        f.write(f"Iterations: {iterations}\n\n")
        
        f.write("ENCODING SPEED:\n")
        f.write(f"BipToken: {results['bip_encoding_time']:.6f} seconds per iteration\n")
        f.write(f"tiktoken: {results['tik_encoding_time']:.6f} seconds per iteration\n")
        if results['tik_encoding_time'] > 0:
            ratio = results['bip_encoding_time'] / results['tik_encoding_time']
            faster = "tiktoken" if ratio > 1 else "BipToken"
            f.write(f"{faster} is {max(ratio, 1/ratio):.2f}x faster at encoding\n\n")
        
        f.write("DECODING SPEED:\n")
        f.write(f"BipToken: {results['bip_decoding_time']:.6f} seconds per iteration\n")
        f.write(f"tiktoken: {results['tik_decoding_time']:.6f} seconds per iteration\n")
        if results['tik_decoding_time'] > 0:
            ratio = results['bip_decoding_time'] / results['tik_decoding_time']
            faster = "tiktoken" if ratio > 1 else "BipToken"
            f.write(f"{faster} is {max(ratio, 1/ratio):.2f}x faster at decoding\n\n")
        
        f.write("PERFECT RECONSTRUCTION:\n")
        f.write(f"BipToken: {'Yes' if results['bip_perfect'] else 'No'}\n")
        f.write(f"tiktoken: {'Yes' if results['tik_perfect'] else 'No'}\n\n")
        
        f.write("TOKEN COUNT:\n")
        f.write(f"BipToken: {results['bip_tokens']} tokens\n")
        f.write(f"tiktoken: {results['tik_tokens']} tokens\n")
    
    print("\nBenchmark completed! Results saved in 'benchmark_results.txt'")