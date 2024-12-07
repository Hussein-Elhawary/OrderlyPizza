import re
import torch
#import nltk 
from transformers import BertTokenizer
from utils import read_test_cases as rtc

def clean_text(text):
    """Remove unwanted characters and symbols from text."""

    pattern = r'(?<="train\.SRC": ").+(?=", "train\.EXR")'
    train_src = re.finditer(pattern, text)
    train_src_arr = []
    for match in train_src:
        train_src_arr.append(match.group())
    return train_src_arr   

# def lemmatize_text(text, lemmatizer = 0):
#     """Lemmatize tokens."""
#     lemmatize_tokens = None

#     # if lemmatizer == 0:
#     #     lemmatizer = spacy.load('en_core_web_sm')
#     #     tokens = lemmatizer(text)
#     #     lemmatized_tokens = [token.lemma_ for token in tokens]
#     # return lemmatized_tokens
#     if lemmatizer == 0:
#         lemmatizer = nltk.stem.WordNetLemmatizer()
#         lemmatize_tokens = [lemmatizer.lemmatize(token) for token in text]
#     return lemmatize_tokens
    

def tokenize_text(text, tokenizer = 0):
    """Tokenize text using a given tokenizer."""
    if tokenizer == 0:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(text)
    return tokens


if __name__ == '__main__':
    text = rtc('./try.json')
    cleaned = clean_text(text)
    #lemmatized = lemmatize_text(cleaned)
    for i in range(len((cleaned))):
        
        #print(lemmatize_text(cleaned[i]))
        print(tokenize_text(cleaned[i]))