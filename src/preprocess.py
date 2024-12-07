import re
from transformers import RobertaTokenizer
from utils import read_test_cases as rtc

def clean_text(text):
    """Remove unwanted characters and symbols from text."""
    pattern = r'(?<="train\.SRC": ").+(?=", "train\.EXR")'
    train_src = re.finditer(pattern, text)
    train_src_arr = []
    for match in train_src:
        train_src_arr.append(match.group())
    return train_src_arr   

def tokenize_text(text, tokenizer):
    """Tokenize text using a given tokenizer."""
    tokens = tokenizer.tokenize(text)
    return tokens

def preprocess_text(path):
    """Preprocess text data."""
    text = rtc(path)
    cleaned = clean_text(text)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenized_text = []
    for i in range(len(cleaned)):
        tokenized_text.append(tokenize_text(cleaned[i], tokenizer))
    return tokenized_text

if __name__ == '__main__':
    text = rtc('./try.json')
    cleaned = clean_text(text)
    print(cleaned)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    for i in range(len(cleaned)):
        print(tokenize_text(cleaned[i], tokenizer))