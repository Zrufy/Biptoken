import unittest
from trutoken import BipTokenizer

class TestBipTokenizer(unittest.TestCase):
    
    def setUp(self):
        # Create a tokenizer for testing
        self.tokenizer = BipTokenizer(vocab_size=1000)
        
        # Minimal training data
        training_data = [
            "Hello world! This is a test.",
            "BipTokenizer preserves spaces   exactly as they appear.",
            "It also preserves UPPERCASE and punctuation!",
            "<user>Special tokens are handled correctly</user>",
            "Numbers like 123.456 are preserved too."
        ]
        
        # Train the tokenizer
        self.tokenizer.train(training_data)
    
    def test_perfect_reconstruction(self):
        """Check that decoding exactly reproduces the original text"""
        test_cases = [
            "Hello world!",
            "This is a NEW test with UPPERCASE.",
            "Multiple   spaces   are   preserved.",
            "<user>Special tokens work too!</user>",
            "Punctuation: !?.,;:()[]{}",
            "Numbers: 42, 3.14159, 1234"
        ]
        
        for text in test_cases:
            ids = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(ids)
            self.assertEqual(text, decoded, f"Failed to reconstruct: '{text}'")
    
    def test_case_preservation(self):
        """Check that uppercase and lowercase are preserved"""
        text = "UPPERCASE lowercase MixedCase"
        ids = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(ids)
        self.assertEqual(text, decoded)
    
    def test_space_preservation(self):
        """Check that multiple spaces are preserved"""
        text = "Single space. Double  space. Triple   space."
        ids = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(ids)
        self.assertEqual(text, decoded)
    
    def test_special_tokens(self):
        """Check that special tokens are handled correctly"""
        text = "<user>Hello</user><assistant>Hi there!</assistant>"
        ids = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(ids)
        self.assertEqual(text, decoded)
    
    def test_save_load(self):
        """Check that saving and loading work correctly"""
        # Save the tokenizer
        self.tokenizer.save("test_tokenizer.json")
        
        # Load into a new tokenizer
        loaded_tokenizer = BipTokenizer()
        loaded_tokenizer.load("test_tokenizer.json")
        
        # Check that it works correctly
        text = "Test after loading"
        ids = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(ids)
        
        loaded_ids = loaded_tokenizer.encode(text)
        loaded_decoded = loaded_tokenizer.decode(loaded_ids)
        
        self.assertEqual(text, decoded)
        self.assertEqual(text, loaded_decoded)
        self.assertEqual(ids, loaded_ids)

if __name__ == "__main__":
    unittest.main()