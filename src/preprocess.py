import re
from transformers import RobertaTokenizer
from datasets import load_dataset, DatasetDict, Dataset
from utils import *
import numpy as np
import os 
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
import random as rnd
import torch
from feature_extraction import *

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
    pattern = r'(?<="train.TOP-DECOUPLED": ").+(?="\})'
    train_top = re.finditer(pattern, text)
    train_top_arr = []
    for match in train_top:
        train_top_arr.append(match.group())
    return train_top_arr

def read_dev_dataset(file_path):

    with open(file_path, 'r') as file:
        text = file.read()
        
    pattern_src = r'(?<="dev\.SRC": ").+(?=", "dev\.EXR")'
    pattern_top = r'(?<="dev\.TOP": ").+(?=", "dev\.PCFG_ERR")'
    test_src = re.finditer(pattern_src, text)
    test_top = re.finditer(pattern_top, text)
    test_src_arr = []
    test_top_decoupled_arr = []

    for match in test_src:
        test_src_arr.append(match.group())
    
    pattern_top_decoupled = r'(?<=\))[\w ]*(?= \()|(?<=ORDER)[\w ]*(?= \()|(?<=PIZZAORDER)[\w ]*(?= \()|(?<=DRINKORDER)[\w ]*(?= \()'
    for match in test_top:
        temp = re.sub(pattern_top_decoupled,'',match.group())
        test_top_decoupled_arr.append(temp)

    return test_src_arr, test_top_decoupled_arr

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
        
        
        if label in ['PIZZAORDER', 'ORDER', 'DRINKORDER']:
            continue
        
        for i in range(len(words)):
            if words[i:i+len(entity_words)] == entity_words:
                bio_tags[i] = f"B-{label}"  
                for j in range(1, len(entity_words)):
                    bio_tags[i+j] = f"I-{label}"  
                break  
    
    return list(zip(words, bio_tags))

def create_test_labels_input():
    longest_sentence = 0
    unique_words = set()
    result = []
    tags = []
    # ut_labels = read_unique_labels('./unique_labels.txt')
    # t_labels = {}
    # t_labels['0'] = 0
    # for i in range(len(ut_labels)):
    #     t_labels[ut_labels[i]] = i+1
    
    test_SRC, test_TOP_DECOUPLED = read_dev_dataset("../dataset/PIZZA_dev.json")
    print(test_SRC[0])
    print(len(test_SRC))
    print(test_TOP_DECOUPLED[0])
    print(len(test_TOP_DECOUPLED))
    test_SRC_size = len(test_SRC)

    with open('../dataset/test_input_labels.txt', 'w') as f:
        for i in range(test_SRC_size):
            test_SRC_item = test_SRC[i]
            test_TOP_DECOUPLED_item = test_TOP_DECOUPLED[i]
            longest_sentence = max(len(test_SRC_item.split()), longest_sentence)    
            unique_words.update(test_SRC_item.split())            
            # print(train_SRC)
            # print(longest_sentence)
            result.append(parse_tc(test_SRC_item,test_TOP_DECOUPLED_item))
            #print(result[i])
            #print("entities above")
            tags.append(generate_bio_tags(result[i]['sentence'], result[i]['entities']))
            #print(tags[i])
            test_SRC_labels_list = []
            for word, tag in tags[i]:
                #print(f"{word}: {tag}")
                test_SRC_labels_list.append(tag)
                #unique_labels.add(tag) if tag != '0' else None
                f.write(f"{tag} ")
            f.write("\n")
            #print("--------------------------------------------------------")
    print(longest_sentence)

def create_word_indices(unique_words):
    """Create a mapping of words to indices."""
    word_to_index = {'<PAD>': 0}
    for word in unique_words:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)
    return word_to_index

def convert_sentences_to_indices(sentences, word_to_index, max_length):
    """Convert sentences to a tensor of word indices and pad them to the same length."""
    indexed_sentences = []
    for sentence in sentences:
        sentence_indices = [word_to_index.get(word, 0) for word in sentence.split()]
        # Pad the sentence to the max_length
        sentence_indices += [word_to_index['<PAD>']] * (max_length - len(sentence_indices))
        indexed_sentences.append(sentence_indices)
    return torch.tensor(indexed_sentences, dtype=torch.long)

class NERDataset(torch.utils.data.Dataset):

  def __init__(self, x, y, max_len):
    """
    This is the constructor of the NERDataset
    Inputs:
    - x: a list of lists where each list contains the ids of the tokens
    - y: a list of lists where each list contains the label of each token in the sentence
    - pad: the id of the <PAD> token (to be used for padding all sentences and labels to have the same length)
    """
    # i guess x should be extended to have the same length as y
    self.x_tensor = x
    self.y_tensor = torch.tensor([seq + [0] * (max_len - len(seq)) for seq in y], dtype=torch.long)
    #################################################################################################################

  def __len__(self):
    """
    This function should return the length of the dataset (the number of sentences)
    """
    ###################### TODO: return the length of the dataset #############################

    return len(self.x_tensor)
  
    ###########################################################################################

  def __getitem__(self, idx):
    """
    This function returns a subset of the whole dataset
    """
    ###################### TODO: return a tuple of x and y ###################################
    return self.x_tensor[idx], self.y_tensor[idx]
    ##########################################################################################

class NER(nn.Module):
    def __init__(self, n_classes, embeddings, hidden_size=50, embedding_dim=768):
        """
        The constructor of our NER model
        Inputs:
        - vacab_size: the number of unique words
        - embedding_dim: the embedding dimension
        - n_classes: the number of final classes (tags)
        """
        super(NER, self).__init__()
        ## Word embedding layer
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding = embeddings
        
        # Combine word and contextual embeddings
        #combined_embedding_dim = embedding_dim + contextual_embedding_dim
        
        # LSTM layer with combined embedding
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        
        # Linear layer
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, embeddings):
        """
        This function does the forward pass of our model
        Inputs:
        - sentences: tensor of shape (batch_size, max_length)

        Returns:
        - final_output: tensor of shape (batch_size, max_length, n_classes)
        """

        # Word embeddings
        #word_embedded = self.embedding(sentences)

        # Ensure contextual embeddings have the same dimensions as word embeddings
        #contextual_embeddings = contextual_embeddings[:, :word_embedded.size(1), :]

        # Concatenate word and contextual embeddings
        #combined_embeddings = torch.cat([word_embedded, contextual_embeddings], dim=-1)
        
        # LSTM and linear layers
        lstm_out, _ = self.lstm(embeddings)
        final_output = self.linear(lstm_out)
        
        return final_output
  
def train(model, train_dataset, batch_size=512, epochs=5, learning_rate=0.01):
    """
    This function implements the training logic
    Inputs:
    - model: the model ot be trained
    - train_dataset: the training set of type NERDataset
    - batch_size: integer represents the number of examples per step
    - epochs: integer represents the total number of epochs (full training pass)
    - learning_rate: the learning rate to be used by the optimizer
    """

    ############################## TODO: replace the Nones in the following code ##################################
    
    # (1) create the dataloader of the training set (make the shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # (2) make the criterion cross entropy loss
    criterion = nn.CrossEntropyLoss()

    # (3) create the optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # GPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        print("CUDA is not available. Training on CPU ...")
    
    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            #print(train_input[0])
            # (4) move the train input to the device
            train_label = train_label.to(device)
            train_input = train_input.to(device)

            # (5) move the train label to the device

            #embeddings = embeddings.to(device)
            # (6) do the forward pass
            output = model(train_input[:, :train_input.size(1), :])
            output = output.permute(0, 2, 1) 
            # print(output.shape,"output")
            # print(train_label.shape,"train_label")
            # print(output.reshape(-1,19).shape,"output.view1")
            # print(output.size(-1),"output.view")
            # print(train_label.view(-1).shape,"train_label.view")
            # (7) loss calculation (you need to think in this part how to calculate the loss correctly)
            batch_loss = criterion(output.reshape(-1,output.size(-1)), train_label.view(-1))

            # (8) append the batch loss to the total_loss_train
            total_loss_train += batch_loss.item()
            
            # (9) calculate the batch accuracy (just add the number of correct predictions)
            acc = torch.sum(torch.argmax(output, dim=-1) == train_label)
            total_acc_train += acc

            # (10) zero your gradients
            optimizer.zero_grad()

            # (11) do the backward pass
            batch_loss.backward()

            # (12) update the weights with your optimizer
            optimizer.step()
        

        ##############################################################################################################    
            # epoch loss
            epoch_loss = total_loss_train / len(train_dataset)

            # (13) calculate the accuracy
            epoch_acc = total_acc_train / (len(train_dataset) * train_dataset.y_tensor.size(1))

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {epoch_loss} \
                | Train Accuracy: {epoch_acc}\n')

if __name__ == '__main__':
    # data_path = "./dataset/source.txt"  
    # try:
    #     data = load_dataset('json', data_files=data_path)
    # except Exception as e:
    #     raise ValueError(f"Failed to load dataset from {data_path}: {e}")
    create_test_labels_input()
    raise KeyError
    train_SRC,_,train_TOP_DECOUPLED = extract_sentences()
    train_SRC = train_SRC[:100]
    train_TOP_DECOUPLED = train_TOP_DECOUPLED[:100]
    
    print(train_SRC[0])
    print(len(train_SRC))

    ut_labels = read_unique_labels('./unique_labels.txt')
    #print("unique labels: ",ut_labels)
    t_labels = {}
    t_labels['0'] = 0
    for i in range(len(ut_labels)):
        t_labels[ut_labels[i]] = i+1
    

    #print("t_labels: ",t_labels)
    train_SRC_size = len(train_SRC)
    result = []
    tags = []
    longest_sentence = 30
    train_SRC_labels = []
    unique_labels = set()
    unique_words = set()
    print("checkpoint 1")
    # with open('input_labels.txt', 'w') as f:
    #     for i in range(train_SRC_size):
    #         train_SRC_item = train_SRC[i]
    #         train_TOP_DECOUPLED_item = train_TOP_DECOUPLED[i]
    #         longest_sentence = max(len(train_SRC_item.split()), longest_sentence)    
    #         unique_words.update(train_SRC_item.split())            
    #         # print(train_SRC)
    #         # print(longest_sentence)
    #         result.append(parse_tc(train_SRC_item,train_TOP_DECOUPLED_item))
    #         #print(result[i])
    #         #print("entities above")
    #         tags.append(generate_bio_tags(result[i]['sentence'], result[i]['entities']))
    #         #print(tags[i])
    #         train_SRC_labels_list = []
    #         for word, tag in tags[i]:
    #             #print(f"{word}: {tag}")
    #             train_SRC_labels_list.append(tag)
    #             #unique_labels.add(tag) if tag != '0' else None
    #             f.write(f"{tag} ")
    #         f.write("\n")
    #         print("--------------------------------------------------------")
    # with open('unique_labels.txt', 'w') as f2:
    #     f2.write("\n".join(unique_labels))
    #train_SRC_data = data['train']['train.SRC']
    co_embeddings = contextual_embeddings(train_SRC)
    #print(co_embeddings[0])
    #co_embeddings = co_embeddings[0]
    # Convert tags to indices
    tag_indices = [[t_labels[tag] for _, tag in sentence_tags] for sentence_tags in tags]
    #print("tags",tags)

    #print("tags: ",tag_indices)
    #train_dataset = NERDataset(co_embeddings, tag_indices, len(ut_labels))

    #assert len(co_embeddings) == len(tag_indices), "Mismatch between number of embeddings and labels"
    #model = NER(len(t_labels),len(unique_words))
    #print(model)
    #print(train_dataset.__getitem__(0))
    #train(model, train_dataset)

    #############################################################################################################
    # In your main script, modify the code:
    unique_words = list(unique_words)
    word_to_index = create_word_indices(unique_words)

    # Convert sentences to word indices
    sentence_indices = convert_sentences_to_indices(train_SRC, word_to_index, longest_sentence)


    # Modified NER initialization
    model = NER(longest_sentence, co_embeddings, hidden_size=50, embedding_dim=768)
    # print("tag_indices",word_to_index)
    # print("sentence_indices",sentence_indices)
    # raise KeyError
    print("longest sentence",longest_sentence)
    # Create dataset with word indices instead of contextual embeddings
    train_dataset = NERDataset(co_embeddings, tag_indices, longest_sentence)    # i think instead of longest sentence it should be len(ut_labels)
    print("train_dataset 0",sentence_indices.shape)
    print("train_dataset 1",tag_indices)
    print(len(unique_words))
    #raise KeyError
    print("checkpoint 3")
    train(model, train_dataset, batch_size=512, epochs=5, learning_rate=0.01)

    ###################################################################################################

  