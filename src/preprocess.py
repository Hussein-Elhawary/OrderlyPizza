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

def preprocess_train_top_decoupled(text):
    """Get the training topics from the text."""
    pattern = r'(?<="train.TOP-DECOUPLED": ").+(?="},)'
    train_top = re.finditer(pattern, text)
    train_top_arr = []
    for match in train_top:
        train_top_arr.append(match.group())
    return train_top_arr

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

def parse_tc(train_SRC,train_TOP):
    train_SRC = "i'd like a pizza with banana pepper grilled chicken and white onions without thin crust"
    train_TOP = "(ORDER i'd like (PIZZAORDER (NUMBER a ) pizza with (TOPPING banana pepper ) (TOPPING grilled chicken ) and (TOPPING white onions ) without (NOT (STYLE thin crust ) ) ) )"

    def parse_sexp(s):
        s = s.replace('(', ' ( ').replace(')', ' ) ')
        tokens = s.split()
        def helper(tokens):
            token = tokens.pop(0)
            if token == '(':
                L = []
                while tokens[0] != ')':
                    L.append(helper(tokens))
                tokens.pop(0)
                return L
            else:
                return token
        return helper(tokens.copy())

    tree = parse_sexp(train_TOP)

    entities = []

    def extract_entities(tree, current_label=None, text_accumulator=[]):
        if isinstance(tree, list):
            label = tree[0]
            content = tree[1:]
            text = []
            for item in content:
                extract_entities(item, label, text)
            entity_text = ' '.join(text)
            if label in ['ORDER', 'PIZZAORDER', 'NOT'] or label not in ['NUMBER']:
                match = re.search(re.escape(entity_text), train_SRC)
                if match:
                    entities.append({
                        'label': label,
                        'word': match.group(),
                    })
            text_accumulator.extend(text)
        else:
            text_accumulator.append(tree)

    extract_entities(tree)

    result = {
        'sentence': train_SRC,
        'entities': entities
    }
    print(result)
    return result

if __name__ == '__main__':
    text = rtc('./test.json')
    cleaned = clean_text(text)
    print(cleaned)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    for i in range(len(cleaned)):
        print(tokenize_text(cleaned[i], tokenizer))

    


