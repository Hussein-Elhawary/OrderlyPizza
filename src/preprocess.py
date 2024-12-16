import re
from transformers import RobertaTokenizer
from datasets import load_dataset, DatasetDict, Dataset
from utils import *
import numpy as np
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
    text = read_test_cases(path)
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
            #if label in ['ORDER', 'PIZZAORDER', 'NOT'] or label not in ['NUMBER']:
            match = re.search(re.escape(entity_text), train_SRC)
            if match:
                if label == "NOT":
                    temp_entity = entities.pop()
                    entities.append({
                    'label': label+"-"+temp_entity['label'],
                    'word': match.group(),
                    })

                else:
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
    #print(result)
    return result

def datasetcreate(train_SRC,train_TOP_DECOUPLED,labels):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    corrected_labels = np.ndarray(len(train_SRC),dtype=object)
    for i in range(len(train_SRC)):
        train_SRC[i] = train_SRC[i].split(" ")
    for i in range(len(train_TOP_DECOUPLED)):
        train_TOP_DECOUPLED[i] = train_TOP_DECOUPLED[i].replace("("," ").split(" ")

    for i in range(len(train_SRC)):
        for j in range(len(train_SRC[i])):
            for k in range(len(train_TOP_DECOUPLED[i])):
                if train_SRC[i][j] == train_TOP_DECOUPLED[i][k]:
                    if train_SRC[i][j-1] == train_TOP_DECOUPLED[i][k-1]:
                        word = train_TOP_DECOUPLED[i][k-2]
                        while labels.get(word) == None:
                            word = train_TOP_DECOUPLED[i][k-2]
                            k = k - 1
                        corrected_labels[i] = np.append(corrected_labels[i],"I-"+str(labels[word]))
                    else:
                        # print(train_SRC[i])
                        # print(train_TOP_DECOUPLED[i])
                        # print(train_SRC[i][j-1])
                        # print(train_TOP_DECOUPLED[i][k])
                        # print(train_TOP_DECOUPLED[i][k-1])
                        corrected_labels[i] = np.append(corrected_labels[i],"B-"+str(labels[train_TOP_DECOUPLED[i][k-1]]))
                    break
            corrected_labels[i] = np.append(corrected_labels[i],"O")

    #for i in range(len(train_SRC)):
    #    train_SRC[i] = tokenize_text(train_SRC[i], tokenizer)
    
    #print(train_SRC)
    #clean_text = tokenize_text(train_SRC, tokenizer)
    #print(train_TOP_DECOUPLED)

def generate_bio_tags(sentence, entities):
    
    words = sentence.split()
    bio_tags = ["0"] * len(words)  
    
    
    for entity in entities:
        label = entity['label'] 
        entity_words = entity['word'].split()  
        
        
        if label in ['PIZZAORDER', 'ORDER']:
            continue
        
        for i in range(len(words)):
            if words[i:i+len(entity_words)] == entity_words:
                bio_tags[i] = f"B-{label}"  
                for j in range(1, len(entity_words)):
                    bio_tags[i+j] = f"I-{label}"  
                break  
    
    return list(zip(words, bio_tags))

if __name__ == '__main__':
    data_path = "./test.json"  
    try:
        data = load_dataset('json', data_files=data_path)
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {data_path}: {e}")
    #print(data['train']['train.TOP'])
    #print(data["train"])
    train_SRC = data['train']['train.SRC']
    train_TOP_DECOUPLED = data['train']['train.TOP-DECOUPLED']
    result = []
    tags = []
    for i in range(len(train_SRC)):
        result.append(parse_tc(train_SRC,train_TOP_DECOUPLED))
        print(result[i])
        print("entities above")
        tags.append(generate_bio_tags(result[i]['sentence'], result[i]['entities']))
        print(tags[i])
        for word, tag in tags[i]:
            print(f"{word}: {tag}")
    
    # labels = read_labels('unique_labels.txt')
    # labels_num = {}
    # for i in range(len(labels)):
    #     labels_num[labels[i]] = i
    # cleaned = datasetcreate(train_SRC,train_TOP_DECOUPLED,labels_num)
    # print(cleaned)
    #tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    #for i in range(len(cleaned)):
    #    print(tokenize_text(cleaned[i], tokenizer))

    


