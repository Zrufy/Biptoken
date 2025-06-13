from trutoken import BipTokenizer

# Basic usage example
def main():
    print("BipTokenizer - Basic usage example")
    print("-" * 50)
    
    # Create a tokenizer
    tokenizer = BipTokenizer(vocab_size=1000)
    
    # Example data for training
    training_data = [
        "Hello world! This is a test.",
        "BipTokenizer preserves spaces   exactly as they appear.",
        "It also preserves UPPERCASE and punctuation!",
        "<user>Special tokens are handled correctly</user>",
        "Numbers like 123.456 are preserved too."
    ]
    
    # Train the tokenizer
    tokenizer.train(training_data)
    
    # Encoding/decoding test
    test_texts = [
        "Hello world!",
        "This is a NEW test with UPPERCASE.",
        "Multiple   spaces   are   preserved.",
        "<user>Special tokens work too!</user>",
        "Punctuation: !?.,;:()[]{}",
        "Numbers: 42, 3.14159, 1234"
    ]
    
    for text in test_texts:
        print("\nOriginal text:", text)
        ids = tokenizer.encode(text)
        print(f"Encoded ({len(ids)} tokens):", ids[:10], "..." if len(ids) > 10 else "")
        decoded = tokenizer.decode(ids)
        print("Decoded:", decoded)
        print("Perfect match:", text == decoded)
    
    # Save and load
    tokenizer.save("example_tokenizer.json")
    print("\nTokenizer saved to 'example_tokenizer.json'")
    
    loaded_tokenizer = BipTokenizer()
    loaded_tokenizer.load("example_tokenizer.json")
    print("Tokenizer loaded successfully")
    
    # Check that the loaded tokenizer works correctly
    text = "Test after loading"
    ids = loaded_tokenizer.encode(text)
    decoded = loaded_tokenizer.decode(ids)
    print(f"\nTest after loading: '{text}' -> '{decoded}'")
    print("Perfect match:", text == decoded)

if __name__ == "__main__":
    main()