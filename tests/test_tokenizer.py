# test_tokenizer.py

from src.data.tokenizer import CharTokenizer

def test_char_tokenizer():
    text = "hello world"
    tokenizer = CharTokenizer(text)
    encoded = tokenizer.encode("hello")
    decoded = tokenizer.decode(encoded)
    assert decoded == "hello"
    assert tokenizer.vocab_size > 0
    print("✅ test_char_tokenizer passed")

if __name__ == "__main__":
    test_char_tokenizer()
    
